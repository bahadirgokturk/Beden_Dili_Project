// Python tarafındaki landmarks.extract_keypoints ve model.SignClassifier'ın tarayıcı karşılığı.
// Hem tarayıcıda (window.TidModel) hem Node'da (require) çalışır; Node ile verify.js doğrular.
(function (root) {
  "use strict";

  // landmarks.py ile aynı: burun, omuzlar, dirsekler, bilekler
  const POSE_IDS = [0, 11, 12, 13, 14, 15, 16];

  function decode(t) {
    const bin = typeof atob === "function"
      ? Uint8Array.from(atob(t.data), (c) => c.charCodeAt(0))
      : Uint8Array.from(Buffer.from(t.data, "base64"));
    return { shape: t.shape, data: new Float32Array(bin.buffer, bin.byteOffset, bin.byteLength / 4) };
  }

  /** MediaPipe Holistic sonucunu 147 elemanlı, omuzlara göre normalleştirilmiş vektöre çevirir. */
  function extractKeypoints(results) {
    const out = new Float32Array(POSE_IDS.length * 3 + 21 * 3 * 2);
    const pose = results.poseLandmarks;
    let cx = 0, cy = 0, cz = 0, scale = 1;
    if (pose) {
      const l = pose[11], r = pose[12];
      cx = (l.x + r.x) / 2; cy = (l.y + r.y) / 2; cz = (l.z + r.z) / 2;
      scale = Math.hypot(l.x - r.x, l.y - r.y) || 1;
    }
    const norm = (p) => (pose ? [(p.x - cx) / scale, (p.y - cy) / scale, (p.z - cz) / scale] : [p.x, p.y, p.z]);
    let i = 0;
    const put = (pts, ids) => {
      for (const id of ids) {
        const v = pts ? norm(pts[id]) : [0, 0, 0];
        out[i++] = v[0]; out[i++] = v[1]; out[i++] = v[2];
      }
    };
    const hand = [...Array(21).keys()];
    put(pose, POSE_IDS);
    put(results.leftHandLandmarks, hand);
    put(results.rightHandLandmarks, hand);
    return out;
  }

  /**
   * Bir kaydı (zaman x 147) çizilebilir iskelet karelerine çevirir: 7 poz noktası, sol ve sağ el.
   * El takibi birkaç kare kaybolduysa son görülen el en fazla 5 kare boyunca tutulur (titremesin diye).
   * fit(W, H) iskeleti tuvale sığdıracak ölçek ve kaydırmayı verir.
   */
  function buildSkeleton(t) {
    const [T, F] = t.shape;
    const P = POSE_IDS.length;
    const pts = (row, start, n) => {
      const out = [];
      let any = false;
      for (let i = 0; i < n; i++) {
        const x = row[start + i * 3], y = row[start + i * 3 + 1];
        if (x !== 0 || y !== 0) any = true;
        out.push([x, y]);
      }
      return any ? out : null;
    };
    const frames = [];
    const last = { left: null, right: null }, age = { left: 99, right: 99 };
    for (let k = 0; k < T; k++) {
      const row = t.data.subarray(k * F, (k + 1) * F);
      const f = { pose: pts(row, 0, P), left: pts(row, P * 3, 21), right: pts(row, P * 3 + 63, 21) };
      for (const side of ["left", "right"]) {
        if (f[side]) { last[side] = f[side]; age[side] = 0; }
        else if (++age[side] <= 5) f[side] = last[side];
      }
      if (f.pose) frames.push(f);
    }
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const f of frames) {
      const sw = Math.hypot(f.pose[1][0] - f.pose[2][0], f.pose[1][1] - f.pose[2][1]);
      const all = [...f.pose, ...(f.left || []), ...(f.right || []),
        [f.pose[0][0], f.pose[0][1] - sw * 0.35]];  // başın üstü
      for (const [x, y] of all) {
        minX = Math.min(minX, x); maxX = Math.max(maxX, x);
        minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      }
    }
    const fit = (W, H) => {
      const m = 0.08;
      const s = Math.min(W * (1 - 2 * m) / (maxX - minX || 1), H * (1 - 2 * m) / (maxY - minY || 1));
      return { s, ox: W / 2 - ((minX + maxX) / 2) * s, oy: H / 2 - ((minY + maxY) / 2) * s };
    };
    return { frames, fit };
  }

  const sigmoid = (x) => 1 / (1 + Math.exp(-x));

  // y = W x + b ; W (rows x cols) satır öncelikli
  function affine(W, b, x, rows, cols, y) {
    for (let r = 0; r < rows; r++) {
      let s = b ? b[r] : 0;
      const off = r * cols;
      for (let c = 0; c < cols; c++) s += W[off + c] * x[c];
      y[r] = s;
    }
    return y;
  }

  // PyTorch GRU tek yön: kapı sırası r, z, n
  function gruDirection(seq, wih, whh, bih, bhh, inSize, H, reverse) {
    const T = seq.length;
    const out = new Array(T);
    let h = new Float32Array(H);
    const gi = new Float32Array(3 * H), gh = new Float32Array(3 * H);
    for (let k = 0; k < T; k++) {
      const t = reverse ? T - 1 - k : k;
      affine(wih, bih, seq[t], 3 * H, inSize, gi);
      affine(whh, bhh, h, 3 * H, H, gh);
      const next = new Float32Array(H);
      for (let j = 0; j < H; j++) {
        const r = sigmoid(gi[j] + gh[j]);
        const z = sigmoid(gi[H + j] + gh[H + j]);
        const n = Math.tanh(gi[2 * H + j] + r * gh[2 * H + j]);
        next[j] = (1 - z) * n + z * h[j];
      }
      h = next;
      out[t] = h;
    }
    return out;
  }

  class SignClassifier {
    constructor(json) {
      this.labels = json.labels;
      this.signs = json.signs;
      this.sequenceLength = json.sequenceLength;
      this.confidenceThreshold = json.confidenceThreshold;
      this.stableFrames = json.stableFrames;
      this.handGrace = json.handGrace ?? 0.6;
      this.fps = json.fps ?? 10;
      this.photos = new Set(json.photos || []);
      this.w = {};
      for (const [k, v] of Object.entries(json.weights)) this.w[k] = decode(v);
      this.skeletons = {};
      for (const [k, v] of Object.entries(json.examples || {})) this.skeletons[k] = buildSkeleton(decode(v));
      this.hidden = this.w["gru.weight_hh_l0"].shape[1];
      this.layers = Object.keys(this.w).filter((k) => /^gru\.weight_ih_l\d+$/.test(k)).length;
    }

    /** seq: Float32Array dizisi (zaman x özellik). Döndürür: her sınıf için olasılık. */
    predict(seq) {
      const H = this.hidden, w = this.w;
      let x = seq;
      for (let l = 0; l < this.layers; l++) {
        const inSize = x[0].length;
        const dir = (sfx, rev) => gruDirection(x, w[`gru.weight_ih_l${l}${sfx}`].data, w[`gru.weight_hh_l${l}${sfx}`].data,
          w[`gru.bias_ih_l${l}${sfx}`].data, w[`gru.bias_hh_l${l}${sfx}`].data, inSize, H, rev);
        const f = dir("", false), b = dir("_reverse", true);
        x = f.map((ft, t) => { const c = new Float32Array(2 * H); c.set(ft); c.set(b[t], H); return c; });
      }
      // zaman ortalaması -> Linear -> ReLU -> Linear
      const mean = new Float32Array(2 * H);
      for (const v of x) for (let j = 0; j < mean.length; j++) mean[j] += v[j] / x.length;
      const h1w = w["head.0.weight"], h2w = w["head.3.weight"];
      const h1 = affine(h1w.data, w["head.0.bias"].data, mean, h1w.shape[0], h1w.shape[1], new Float32Array(h1w.shape[0]));
      for (let j = 0; j < h1.length; j++) h1[j] = Math.max(0, h1[j]);
      const logits = affine(h2w.data, w["head.3.bias"].data, h1, h2w.shape[0], h2w.shape[1], new Float32Array(h2w.shape[0]));
      const m = Math.max(...logits);
      const e = Array.from(logits, (v) => Math.exp(v - m));
      const s = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / s);
    }
  }

  const api = { extractKeypoints, SignClassifier, decode, buildSkeleton, POSE_IDS };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.TidModel = api;
})(typeof window !== "undefined" ? window : globalThis);
