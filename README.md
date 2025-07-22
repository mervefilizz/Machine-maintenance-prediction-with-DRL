# 🛠️ Derin Pekiştirmeli Öğrenme ile Makine Bakım Tahmini

Bu proje, **Derin Pekiştirmeli Öğrenme (Deep Reinforcement Learning - DRL)** yöntemi kullanılarak makineler için kestirimci bakım zamanlarını otomatik olarak belirlemeyi amaçlamaktadır. Özellikle **Sanayi 4.0** ve **IIoT** yaklaşımları doğrultusunda, sensör verileri ile desteklenen akıllı bir bakım kararı mekanizması oluşturulmuştur.

---

## 🔍 Problem Tanımı

Kestirimci bakım (PdM), makinelerin arıza riski değerlendirilerek optimum bakım zamanlaması yapılmasını hedefler. Bu projede bu karar süreci, bir **pekiştirmeli öğrenme** problemi olarak ele alınmış; ajan her zaman adımında "bakım yap" veya "yapma" kararını alarak uzun vadeli ödülünü maksimize etmeyi öğrenmiştir.

---

## 📦 Kullanılan Veriler

Projede aşağıdaki 5 veri kümesi kullanılmıştır:

- `telemetry.csv`: Zaman serisi verileri (sıcaklık, basınç vs.)
- `failures.csv`: Gerçekleşen arıza zamanları
- `maint.csv`: Gerçekleşen bakım zamanları
- `errors.csv`: Hata türleri ve zamanları
- `machines.csv`: Makine sabit bilgileri (örneğin yaş)

Veriler zaman bazlı birleştirilerek ajan için gözlem vektörleri oluşturulmuştur.

---

## 🌍 Ortam Tasarımı (`PdMEnv`)

Ortam, `pdmenv.py` dosyasındaki **`PdMEnv`** sınıfı ile oluşturulmuş ve **Gymnasium** arayüzü ile uyumlu hale getirilmiştir.

- **Gözlem Uzayı**: Son 24 zaman adımındaki veriler (telemetri, hata, bakım, arıza, yaş, son bakım süresi)
- **Eylem Uzayı**:  
  - `0`: Bakım yapma  
  - `1`: Bakım yap
- **Ödül Fonksiyonu**:
  - Bakım yapılırsa: `-10`
  - Arıza gerçekleşirse ve bakım yapılmadıysa: `-100`
  - Sorunsuz çalıştıysa: `+1`

---

## 🧠 Ajan Mimarisi (DQN)

Ajan, **Deep Q-Network (DQN)** kullanılarak geliştirilmiştir (`DRLTrain.py` dosyasında).

### Ağ Yapısı:
- Giriş: Gözlem vektörü (örn. 24 × N boyutlu)
- Gizli Katman: 64 nöron, ReLU aktivasyonu
- Çıkış Katmanı: 2 eylem

### Öğrenme Parametreleri:
- Gamma (γ): `0.99`
- Öğrenme oranı: `1e-4`
- Batch boyutu: `32`
- Replay buffer: `10,000`
- Epsilon-greedy keşif stratejisi

---

## 🏋️ Eğitim Süreci

Ajan 1000 bölüm boyunca eğitilmiştir:

- Her 1000 adımda hedef ağ güncellenmiştir
- TensorBoard ile kayıplar ve ödüller izlenmiştir
- Eğitim sonunda model `dqn_pdm.pth` dosyasına kaydedilmiştir

Eğitim sırasında ajan, giderek daha stratejik kararlar almaya başlamıştır.

---

## ⚖️ Q-Learning ile Karşılaştırma

| Özellik                  | Q-learning                    | DQN                          |
|--------------------------|-------------------------------|-------------------------------|
| Durum temsili            | Tuple                         | Sürekli tensör               |
| Genelleme yeteneği       | Yok                           | Var                          |
| Bellek kullanımı         | Yüksek                        | Verimli                      |
| Eğitim süresi            | Hızlı                         | Daha uzun                    |
| Performans               | Orta                          | Uzun vadede daha iyi         |

Klasik Q-learning, yüksek boyutlu sürekli veriyle verimsiz kalırken; DQN genelleme yeteneği ile daha iyi performans göstermiştir.

---

## ✅ Sonuçlar

- Ajan, bakım kararlarını **daha stratejik** şekilde vermeyi öğrenmiştir.
- Arıza sayısında azalma, bakım maliyetlerinde optimizasyon sağlanmıştır.
- Sabit aralıklı bakım stratejilerinden daha üstün sonuçlar elde edilmiştir.

---

## 📂 Dosya Yapısı

```text
├── DRLTrain.py               # Eğitim scripti
├── pdmenv.py                 # Gym ortam tanımı
├── 24435004049_Merve_Filiz_Rapor.pdf  # Proje raporu
├── telemetry.csv             # Zaman serisi verileri
├── failures.csv              # Arıza zamanları
├── maint.csv                 # Bakım kayıtları
├── errors.csv                # Hata türleri
├── machines.csv              # Makine bilgileri
└── dqn_pdm.pth               # Eğitilmiş model
