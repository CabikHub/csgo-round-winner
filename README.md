# 🎮 CS:GO Round Winner Prediction
### CT vs T — Makine Öğrenmesi ile E-Spor Strateji Motoru

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-green?style=for-the-badge)

**YBS3259 | Makine Öğrenmesi — Bahar Dönemi Final Projesi**

</div>

---

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Problem Tanımı](#-problem-tanımı)
- [Veri Seti](#-veri-seti)
- [Metodoloji — CRISP-DM](#-metodoloji--crisp-dm)
- [Kullanılan Modeller](#-kullanılan-modeller)
- [Başarı Metrikleri](#-başarı-metrikleri)
- [Proje Yapısı](#-proje-yapısı)
- [Kurulum ve Çalıştırma](#-kurulum-ve-çalıştırma)
- [Riskler ve Kısıtlar](#-riskler-ve-kısıtlar)
- [Proje Ekibi](#-proje-ekibi)

---

## 🎯 Proje Hakkında

Bu proje, **Counter-Strike: Global Offensive (CS:GO)** oyununda bir round'un belirli bir anındaki anlık oyun durumunu analiz ederek, o round'u **hangi takımın kazanacağını (CT mi, T mi?)** öngören bir **makine öğrenmesi tabanlı sınıflandırma motoru** geliştirmeyi amaçlamaktadır.

Takım ekonomisi (eco round, force buy kararları), silah donanımları, anlık sağlık/zırh durumu ve harita dinamiklerini bir arada değerlendiren bu model; **koçlar ve analistler için gerçek zamanlı bir karar destek aracı** işlevi görebilir.

> 💡 **Neden bu proje?**
> E-spor, milyarlarca dolarlık bir endüstri haline gelmiştir. CS:GO'da koçların sezgisel yaptığı "bu round'u kazanabilir miyiz?" analizini, veri odaklı ve ölçülebilir bir mekanizmaya dönüştürmek, hem akademik hem de pratik değer taşımaktadır.

---

## 🔍 Problem Tanımı

| | |
|---|---|
| **Problem Türü** | İkili Sınıflandırma (Binary Classification) |
| **Hedef Değişken** | `round_winner` — CT veya T |
| **Sınıf Dengesi** | ~%50 / %50 (mükemmel dengeli) |
| **Kullanım Amacı** | Anlık oyun durumundan raund galibi tahmini |

### Kazanma Koşulları

**🛡️ CT (Counter-Terrorist) Kazanır:**
- Tüm T oyuncuları öldürülürse
- Bomba patlatılmadan süre dolarsa
- Yerleştirilen bomba defuse edilirse

**💣 T (Terrorist) Kazanır:**
- Tüm CT oyuncuları öldürülürse
- Bomba başarıyla patlarsa

---

## 📊 Veri Seti

| Özellik | Değer |
|---|---|
| **Kaynak Dosya** | `YBS3259_CSGO_Final_Verisi.csv` |
| **Özellik Sayısı** | 96 özellik (feature) |
| **Hedef Değişken** | `round_winner` |
| **Veri Türü** | Profesyonel e-spor maç kayıtları |

### Temel Özellik Grupları

```
📦 Ekonomi Verileri      → Takım bütçesi, silah maliyetleri
🔫 Silah Donanımı        → CT ve T tarafındaki silah inventory'si
❤️  Sağlık & Zırh        → Tüm oyuncuların anlık HP ve armor değerleri
🗺️  Harita Bilgisi        → Oynanan harita (map encoding ile)
👥 Oyuncu Sayısı         → Canlı oyuncu sayıları (her iki takım için)
```

---

## 🔬 Metodoloji — CRISP-DM

Proje, endüstri standardı **CRISP-DM (Cross-Industry Standard Process for Data Mining)** metodolojisi ile 6 fazda yürütülmüştür:

```
FAZ 1 → Business Understanding    📌 İş probleminin tanımlanması
FAZ 2 → Data Understanding        🔍 Veri keşfi ve EDA
FAZ 3 → Data Preparation          🛠️  Temizleme, encoding, feature engineering
FAZ 4 → Modeling                  🤖 Model eğitimi ve karşılaştırma
FAZ 5 → Evaluation                📈 Performans değerlendirmesi
FAZ 6 → Deployment                🚀 Uygulama ve raporlama
```

---

## 🤖 Kullanılan Modeller

Birden fazla algoritma eğitilerek karşılaştırmalı analiz yapılmıştır:

| Model | Tür |
|---|---|
| **Logistic Regression** | Temel sınıflandırma |
| **Decision Tree** | Kural tabanlı |
| **Random Forest** | Ensemble (Bagging) |
| **Gradient Boosting** | Ensemble (Boosting) |
| **XGBoost** | Ensemble (Boosting) |
| **K-Nearest Neighbors (KNN)** | Instance-based |
| **SVM** | Kernel tabanlı |

### Pipeline Mimarisi

```
Ham Veri → Preprocessing (Scaling + Encoding) → Model Eğitimi → Tahmin
```

- **StandardScaler** ile sayısal özellikler ölçeklendirilmiştir
- **Encoding** ile kategorik değişkenler (harita vb.) dönüştürülmüştür
- **Cross-Validation** ile overfitting riski kontrol altına alınmıştır

---

## 📈 Başarı Metrikleri

Sadece Accuracy odaklı değil; **her iki takıma da adil ve dengeli** bir performans hedeflenmiştir. Bu nedenle temel metrik olarak **Macro F1-Score** seçilmiştir.

| Metrik | Hedef |
|---|---|
| **Accuracy** | ≥ %75 |
| **Macro F1-Score** | ≥ %73 |
| **ROC-AUC** | Yüksek |
| **Precision (CT)** | Dengeli |
| **Recall (CT)** | Dengeli |
| **Cross-Val Std** | Düşük (Overfitting yok) |

> **Neden Macro F1?** CT veya T'ye sadece odaklanmak yerine her iki sınıf için de tutarlı performans elde etmek hedeflenmiştir. Yanlış tahminlerin oyun içi ekonomik kararlar üzerindeki maliyeti minimize edilmek istenmiştir.

---

## 🗂️ Proje Yapısı

```
final-project/
├── data/
│   ├── raw/                  # Orijinal veri seti
│   └── processed/            # Temizlenmiş ve işlenmiş veri
├── notebooks/
│   └── CSGO_FİNAL.ipynb      # Ana CRISP-DM analiz notebook'u
├── models/
│   ├── best_model.joblib     # En iyi model (serialize edilmiş)
│   └── pipeline.joblib       # Preprocessing pipeline
├── app/
│   ├── streamlit_app.py      # Streamlit arayüzü
│   └── api.py                # FastAPI endpoint (opsiyonel)
├── figures/                  # Grafik çıktıları
├── requirements.txt          # Paket listesi
├── README.md                 # Bu dosya
└── report.pdf                # Final danışmanlık raporu
```

---

## ⚙️ Kurulum ve Çalıştırma

### 1. Repoyu Klonla

```bash
git clone https://github.com/CabikHub/csgo-round-winner.git
cd csgo-round-winner
```

### 2. Sanal Ortam Oluştur

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Gereksinimleri Yükle

```bash
pip install -r requirements.txt
```

### 4. Notebook'u Çalıştır

```bash
jupyter notebook notebooks/CSGO_FİNAL.ipynb
```

### 5. Streamlit Uygulamasını Başlat (Opsiyonel)

```bash
streamlit run app/streamlit_app.py
```

### Gerekli Kütüphaneler

```
pandas
numpy
scikit-learn
xgboost
plotly
jupyter
streamlit
fastapi
joblib
```

---

## ⚠️ Riskler ve Kısıtlar

| Risk | Açıklama | Önlem |
|---|---|---|
| **Data Leakage** | Round bitişine yakın veri modelin "kopya çekmesine" neden olabilir | Zaman filtresi uygulandı |
| **Overfitting** | 96 özellik ile ezber riski yüksek | Cross-validation + regularization |
| **Harita Etkisi** | Farklı haritalarda farklı dinamikler geçerli | `map` değişkeni encoding ile modele dahil edildi |
| **Oyun Güncellemeleri** | Valve'ın patch'leri silah dengesini bozabilir | Model periyodik güncelleme gerektirir |
| **Amatör vs Profesyonel** | Model profesyonel maçlarla eğitildi | Matchmaking verilerine doğrudan uygulanamaz |

---

## 👥 Proje Ekibi

| İsim | Rol |
|---|---|
| **Emirhan Çabık** | Veri analizi, modelleme |
| **Tevfik Özyurt** | Feature engineering, değerlendirme |
| **Mücahit Selman Şahin** | Görselleştirme, raporlama |

---

## 📚 Ders Bilgisi

```
Ders    : YBS3259 — Makine Öğrenmesi
Dönem   : Bahar Dönemi — Final Projesi
Yöntem  : CRISP-DM
```

---

<div align="center">

**🛡️ CT** &nbsp;|&nbsp; **💣 T** &nbsp;|&nbsp; Hangi taraf kazanıyor? Veri bilir.

*Made with ❤️ and data*

</div>
