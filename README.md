# TİD Altyazı Demosu

Türk İşaret Dili (TİD) işaretlerini tarayıcıda kameradan tanıyıp Türkçe, İngilizce,
Almanca ve Fransızca altyazıya çeviren prototip.

**Canlı demo:** https://bahadirgokturk.github.io/Beden_Dili_Project/

- 9 havalimanı işareti: merhaba, teşekkürler, yardım, su, tuvalet, bilet, uçak, ağrı, hayır
- El ve vücut takibi MediaPipe Holistic ile, sınıflandırma küçük bir GRU modeliyle yapılır.
- Görüntü yalnızca kullanıcının cihazında işlenir, hiçbir sunucuya gönderilmez.
- Bu ilk prototip tek bir kişinin kayıtlarıyla eğitildi; farklı kişilerde doğruluk düşebilir.

| Dosya | Görevi |
|---|---|
| `index.html` | Demo sayfası |
| `tid-model.js` | Nokta çıkarımı ve modelin tarayıcı sürümü |
| `model.json` | Eğitilmiş model ağırlıkları |
