# TİD Altyazı Demosu

Türk İşaret Dili (TİD) işaretlerini tarayıcıda kameradan tanıyıp Türkçe, İngilizce,
Almanca ve Fransızca altyazıya çeviren araştırma prototipi.

**Canlı demo:** https://bahadirgokturk.github.io/Beden_Dili_Project/

- 30 havalimanı işareti: merhaba, selam, hoşça kal, teşekkürler, özür dilerim, lütfen, tamam,
  evet, hayır, yardım, su, aç, içmek, tuvalet, bilet, uçak, kapı, saat, acele, bekle, nerede,
  kimlik, para, telefon, yasak, ağrı, hasta, doktor, ilaç, hastane
- El ve vücut takibi MediaPipe Holistic ile, sınıflandırma küçük bir GRU modeliyle yapılır.
- Görüntü yalnızca kullanıcının cihazında işlenir, hiçbir sunucuya gönderilmez.

## Veri ve atıf

Model, kendi kayıtlarımızın yanında AUTSL veri setiyle de eğitildi:

> O. M. Sincan, H. Y. Keleş. *AUTSL: A Large Scale Multi-Modal Turkish Sign Language Dataset
> and Baseline Methods.* IEEE Access, 2020. https://arxiv.org/abs/2008.00932

AUTSL akademik araştırma amaçlı bir veri setidir; bu depo ve demo ticari olmayan bir araştırma
prototipidir. Depoda AUTSL'in videoları ya da ham verisi bulunmaz.

| Dosya | Görevi |
|---|---|
| `index.html` | Demo sayfası |
| `tid-model.js` | Nokta çıkarımı ve modelin tarayıcı sürümü |
| `model.json` | Eğitilmiş model ağırlıkları |
