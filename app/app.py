# ============================================================
# CS2 Round Winner Predictor — PRO DASHBOARD VERSİYONU
# Pipeline: Faz3 (Feature Eng.) + Faz4 (Extra Trees) + SHAP + UI
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="CS2 Pro Predictor", page_icon="🔫", layout="wide", initial_sidebar_state="expanded")

# --- ÖZEL CSS TEMA (STEAM & CS2 KONSEPTİ) ---
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;600;800&display=swap');
  
  html, body, [class*="css"] { 
      background-color: #0b0e14 !important; /* CS2 Koyu Tema */
      color: #e2e8f0 !important; 
      font-family: 'Inter', sans-serif;
  }
  .stApp { background: #0b0e14; }
  
  /* Başlık ve Metinler */
  h1, h2, h3 { font-family: 'Rajdhani', sans-serif !important; text-transform: uppercase; letter-spacing: 1px; }
  
  /* Sidebar */
  [data-testid="stSidebar"] { 
      background: linear-gradient(180deg, #131722 0%, #0b0e14 100%) !important; 
      border-right: 1px solid #1f2937 !important; 
  }
  
  /* Butonlar */
  .stButton > button {
      background: linear-gradient(90deg, #d97706, #b45309) !important; /* CS2 Turuncu/Gold */
      color: white !important; border: 1px solid #f59e0b !important; border-radius: 4px !important;
      font-family: 'Rajdhani', sans-serif !important; font-weight: 700 !important; font-size: 1.2rem !important;
      padding: 0.5rem 1rem !important; width: 100% !important; transition: all 0.3s ease;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
  }
  .stButton > button:hover { background: linear-gradient(90deg, #f59e0b, #d97706) !important; transform: translateY(-2px); box-shadow: 0 0 15px rgba(217,119,6,0.5) !important; }
  
  /* Metrikler */
  [data-testid="metric-container"] {
      background: #111827 !important; border: 1px solid #374151 !important; 
      border-left: 4px solid #d97706 !important; border-radius: 6px !important; padding: 15px !important;
  }
  [data-testid="metric-container"] label { color: #9ca3af !important; font-size: 0.85rem !important; text-transform: uppercase;}
  [data-testid="metric-container"] [data-testid="metric-value"] { color: #f3f4f6 !important; font-family: 'Rajdhani', sans-serif !important; font-size: 2rem !important; }
  
  /* Sekmeler (Tabs) */
  .stTabs [data-baseweb="tab-list"] { background-color: transparent; gap: 8px; }
  .stTabs [data-baseweb="tab"] { 
      background-color: #1f2937; border-radius: 4px 4px 0 0; padding: 10px 20px; 
      border: 1px solid #374151; border-bottom: none; color: #9ca3af; 
  }
  .stTabs [aria-selected="true"] { 
      background-color: #d97706 !important; color: white !important; border-color: #d97706 !important; 
  }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE BAŞLATMA (HIZLI SENARYOLAR İÇİN) ---
# Kullanıcının arayüzde girdiği değerleri anlık olarak sıfırlamak veya değiştirmek için
default_state = {
    'time_left': 114.0, 'ct_score': 8, 't_score': 7,
    'ct_money': 16000, 't_money': 14000, 'ct_health': 500, 't_health': 500,
    'ct_armor': 500, 't_armor': 400, 'ct_alive': 5, 't_alive': 5,
    'ct_helmets': 5, 't_helmets': 4, 'ct_defuse_kits': 3
}

# Silah listeleri
weapons_ct_rifles = ['ct_weapon_m4a4', 'ct_weapon_m4a1s', 'ct_weapon_aug', 'ct_weapon_famas', 'ct_weapon_ak47']
weapons_t_rifles = ['t_weapon_ak47', 't_weapon_sg553', 't_weapon_galilar', 't_weapon_m4a4']
weapons_snipers = ['ct_weapon_awp', 't_weapon_awp', 'ct_weapon_ssg08', 't_weapon_ssg08', 'ct_weapon_scar20', 't_weapon_g3sg1']
weapons_pistols = ['ct_weapon_usps', 't_weapon_glock', 'ct_weapon_deagle', 't_weapon_deagle', 'ct_weapon_p250', 't_weapon_p250', 'ct_weapon_fiveseven', 't_weapon_tec9', 'ct_weapon_cz75auto', 't_weapon_cz75auto', 'ct_weapon_elite', 't_weapon_elite']
grenades = ['ct_grenade_flashbang', 't_grenade_flashbang', 'ct_grenade_smokegrenade', 't_grenade_smokegrenade', 'ct_grenade_hegrenade', 't_grenade_hegrenade', 'ct_grenade_incendiarygrenade', 't_grenade_molotovgrenade']
other_weapons = ['ct_weapon_mac10', 't_weapon_mac10', 'ct_weapon_mp9', 't_weapon_mp9', 'ct_weapon_ump45', 't_weapon_ump45', 'ct_weapon_p90', 't_weapon_p90', 'ct_weapon_mp5sd', 't_weapon_mp5sd', 'ct_weapon_mp7', 't_weapon_mp7', 't_weapon_bizon', 'ct_weapon_nova', 't_weapon_nova', 'ct_weapon_xm1014', 't_weapon_xm1014', 'ct_weapon_mag7', 't_weapon_sawedoff', 'ct_weapon_m249', 'ct_grenade_decoygrenade', 't_grenade_decoygrenade', 't_weapon_r8revolver']

all_dynamic_keys = list(default_state.keys()) + weapons_ct_rifles + weapons_t_rifles + weapons_snipers + weapons_pistols + grenades + other_weapons

for key in all_dynamic_keys:
    if key not in st.session_state:
        st.session_state[key] = default_state.get(key, 0) # Silahlar varsayılan 0

# Senaryo Fonksiyonları
def set_scenario(scenario_name):
    if scenario_name == "Pistol Raundu":
        st.session_state.update({
            'time_left': 115.0, 'ct_money': 800, 't_money': 800, 'ct_health': 500, 't_health': 500,
            'ct_armor': 0, 't_armor': 0, 'ct_helmets': 0, 't_helmets': 0, 'ct_defuse_kits': 1,
            'ct_weapon_usps': 5, 't_weapon_glock': 5
        })
        for w in weapons_ct_rifles + weapons_t_rifles + weapons_snipers + grenades: st.session_state[w] = 0
    elif scenario_name == "Full Buy (İki Takım)":
        st.session_state.update({
            'ct_money': 25000, 't_money': 28000, 'ct_health': 500, 't_health': 500,
            'ct_armor': 500, 't_armor': 500, 'ct_helmets': 5, 't_helmets': 5, 'ct_defuse_kits': 4,
            'ct_weapon_m4a4': 3, 'ct_weapon_awp': 1, 'ct_weapon_m4a1s': 1, 'ct_grenade_flashbang': 5, 'ct_grenade_smokegrenade': 4,
            't_weapon_ak47': 4, 't_weapon_awp': 1, 't_grenade_flashbang': 6, 't_grenade_smokegrenade': 5, 't_grenade_molotovgrenade': 3
        })

# --- YAN MENÜ (SİDEBAR) ---
with st.sidebar:
    st.image("https://cdn.akamai.steamstatic.com/steam/apps/730/header.jpg", use_container_width=True)
    st.markdown("### ⚙️ Hızlı Senaryolar")
    st.markdown("Sunum sırasında jüriye hızlıca test senaryoları göstermek için:")
    
    if st.button("🔫 Pistol Raundu (1. Raund)"): set_scenario("Pistol Raundu")
    if st.button("💰 Full Buy (Tam Teçhizat)"): set_scenario("Full Buy (İki Takım)")
    
    st.markdown("---")
    st.markdown("### 🗺️ Genel Ayarlar")
    selected_map = st.selectbox("Oynanan Harita", ["de_dust2", "de_mirage", "de_inferno", "de_nuke", "de_overpass", "de_vertigo", "de_train", "de_cache"])
    bomb_planted = st.checkbox("💣 Bomba Kuruldu mu?", value=False)
    
    st.markdown("---")
    st.caption("Veri Analisti Modülü v2.0")

# --- MODEL YÜKLEME ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists(): return None
    return joblib.load(MODEL_PATH)

pipeline = load_model()

if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- ANA EKRAN BAŞLIĞI ---
st.markdown("""
<div style="background:linear-gradient(135deg, #1f2937, #111827); border-left: 5px solid #d97706;
border-radius:8px; padding:20px 30px; margin-bottom:24px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
  <div style="font-family:'Rajdhani', sans-serif; font-size:0.9rem; color:#9ca3af; letter-spacing:0.1em;">
    MAKİNE ÖĞRENMESİ TABANLI ROUND ANALİZ SİSTEMİ
  </div>
  <h1 style="margin:0; font-size:2.5rem; color:#f8fafc; text-shadow: 0 2px 4px rgba(0,0,0,0.5);">CS2 Round Winner Predictor</h1>
</div>
""", unsafe_allow_html=True)

if pipeline is None:
    st.error("❌ `models/best_model.joblib` bulunamadı. Lütfen model dosyanızı `models/` klasörüne ekleyin.")
    st.stop()

# --- SEKMELER (TABS) ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Temel İstatistikler", "🔵 CT Envanteri", "🔴 T Envanteri", "📈 Model Performansı & Metrikler"])

with tab1:
    st.markdown("### ⏱️ Raund Durumu")
    st.session_state['time_left'] = st.slider("Kalan Süre (Saniye)", 0.0, 175.0, st.session_state['time_left'])
    
    col_ct, col_vs, col_t = st.columns([2, 0.5, 2])
    
    with col_ct:
        st.markdown("<h3 style='color: #60a5fa;'>🔵 COUNTER-TERRORISTS</h3>", unsafe_allow_html=True)
        st.session_state['ct_score'] = st.number_input("Skor", 0, 15, st.session_state['ct_score'], key="ct_s")
        st.session_state['ct_alive'] = st.number_input("Hayatta Kalan", 0, 5, st.session_state['ct_alive'], key="ct_al")
        st.session_state['ct_money'] = st.number_input("Toplam Para ($)", 0, 80000, st.session_state['ct_money'], step=500, key="ct_m")
        st.session_state['ct_health'] = st.number_input("Toplam HP", 0, 500, st.session_state['ct_health'], key="ct_h")
        st.session_state['ct_armor'] = st.number_input("Toplam Zırh", 0, 500, st.session_state['ct_armor'], key="ct_ar")
        st.session_state['ct_helmets'] = st.number_input("Kask Sayısı", 0, 5, st.session_state['ct_helmets'], key="ct_helm")
        st.session_state['ct_defuse_kits'] = st.number_input("Çözme Kiti", 0, 5, st.session_state['ct_defuse_kits'], key="ct_def")

    with col_vs:
        st.markdown("<h1 style='text-align:center; color:#6b7280; margin-top: 100px;'>VS</h1>", unsafe_allow_html=True)

    with col_t:
        st.markdown("<h3 style='color: #f87171;'>🔴 TERRORISTS</h3>", unsafe_allow_html=True)
        st.session_state['t_score'] = st.number_input("Skor", 0, 15, st.session_state['t_score'], key="t_s")
        st.session_state['t_alive'] = st.number_input("Hayatta Kalan", 0, 5, st.session_state['t_alive'], key="t_al")
        st.session_state['t_money'] = st.number_input("Toplam Para ($)", 0, 80000, st.session_state['t_money'], step=500, key="t_m")
        st.session_state['t_health'] = st.number_input("Toplam HP", 0, 500, st.session_state['t_health'], key="t_h")
        st.session_state['t_armor'] = st.number_input("Toplam Zırh", 0, 500, st.session_state['t_armor'], key="t_ar")
        st.session_state['t_helmets'] = st.number_input("Kask Sayısı", 0, 5, st.session_state['t_helmets'], key="t_helm")

with tab2:
    st.markdown("### 🔵 Counter-Terrorist Silah & Teçhizat Paneli")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 💥 Tüfekler")
        st.session_state['ct_weapon_m4a4'] = st.number_input("M4A4", 0, 5, st.session_state['ct_weapon_m4a4'])
        st.session_state['ct_weapon_m4a1s'] = st.number_input("M4A1-S", 0, 5, st.session_state['ct_weapon_m4a1s'])
        st.session_state['ct_weapon_aug'] = st.number_input("AUG", 0, 5, st.session_state['ct_weapon_aug'])
        st.session_state['ct_weapon_awp'] = st.number_input("AWP (CT)", 0, 5, st.session_state['ct_weapon_awp'])
        st.session_state['ct_weapon_ak47'] = st.number_input("Yerden Alınan AK-47", 0, 5, st.session_state['ct_weapon_ak47'])
    with c2:
        st.markdown("#### 🔫 Tabancalar")
        st.session_state['ct_weapon_usps'] = st.number_input("USP-S", 0, 5, st.session_state['ct_weapon_usps'])
        st.session_state['ct_weapon_deagle'] = st.number_input("Desert Eagle", 0, 5, st.session_state['ct_weapon_deagle'])
        st.session_state['ct_weapon_p250'] = st.number_input("P250", 0, 5, st.session_state['ct_weapon_p250'])
        st.session_state['ct_weapon_fiveseven'] = st.number_input("Five-SeveN", 0, 5, st.session_state['ct_weapon_fiveseven'])
    with c3:
        st.markdown("#### 💣 Bombalar")
        st.session_state['ct_grenade_smokegrenade'] = st.number_input("Sis Bombası", 0, 5, st.session_state['ct_grenade_smokegrenade'])
        st.session_state['ct_grenade_flashbang'] = st.number_input("Kör Eden Bomba", 0, 10, st.session_state['ct_grenade_flashbang'])
        st.session_state['ct_grenade_hegrenade'] = st.number_input("El Bombası", 0, 5, st.session_state['ct_grenade_hegrenade'])
        st.session_state['ct_grenade_incendiarygrenade'] = st.number_input("Yanıcı Bomba", 0, 5, st.session_state['ct_grenade_incendiarygrenade'])

with tab3:
    st.markdown("### 🔴 Terrorist Silah & Teçhizat Paneli")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("#### 💥 Tüfekler")
        st.session_state['t_weapon_ak47'] = st.number_input("AK-47", 0, 5, st.session_state['t_weapon_ak47'])
        st.session_state['t_weapon_sg553'] = st.number_input("SG 553", 0, 5, st.session_state['t_weapon_sg553'])
        st.session_state['t_weapon_galilar'] = st.number_input("Galil AR", 0, 5, st.session_state['t_weapon_galilar'])
        st.session_state['t_weapon_awp'] = st.number_input("AWP (T)", 0, 5, st.session_state['t_weapon_awp'])
        st.session_state['t_weapon_m4a4'] = st.number_input("Yerden Alınan M4A4", 0, 5, st.session_state['t_weapon_m4a4'])
    with t2:
        st.markdown("#### 🔫 Tabancalar")
        st.session_state['t_weapon_glock'] = st.number_input("Glock-18", 0, 5, st.session_state['t_weapon_glock'])
        st.session_state['t_weapon_deagle'] = st.number_input("Desert Eagle (T)", 0, 5, st.session_state['t_weapon_deagle'])
        st.session_state['t_weapon_tec9'] = st.number_input("Tec-9", 0, 5, st.session_state['t_weapon_tec9'])
        st.session_state['t_weapon_p250'] = st.number_input("P250 (T)", 0, 5, st.session_state['t_weapon_p250'])
    with t3:
        st.markdown("#### 💣 Bombalar")
        st.session_state['t_grenade_smokegrenade'] = st.number_input("Sis Bombası (T)", 0, 5, st.session_state['t_grenade_smokegrenade'])
        st.session_state['t_grenade_flashbang'] = st.number_input("Kör Eden Bomba (T)", 0, 10, st.session_state['t_grenade_flashbang'])
        st.session_state['t_grenade_hegrenade'] = st.number_input("El Bombası (T)", 0, 5, st.session_state['t_grenade_hegrenade'])
        st.session_state['t_grenade_molotovgrenade'] = st.number_input("Molotof", 0, 5, st.session_state['t_grenade_molotovgrenade'])

with tab4:
    st.markdown("### 🤖 Makine Öğrenmesi Model Metrikleri")
    st.markdown("Bu sekme, modelin eğitim aşamasındaki (Deployment öncesi) test başarısını ve karar mimarisini özetler.")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric(label="Model Algoritması", value="Extra Trees") # BURASI GÜNCELLENDİ
    
    # DİKKAT KİRVE: Buradaki Accuracy ve ROC-AUC değerlerini kendi modeline göre değiştirmeyi unutma!
    col_m2.metric(label="Test Accuracy", value="%84.5") 
    col_m3.metric(label="ROC-AUC Skoru", value="0.912")
    
    st.markdown("---")
    st.markdown("#### 📌 En Önemli Değişkenler (Feature Importance)")
    st.markdown("Modelin Round kazananını belirlerken en çok dikkat ettiği 5 özellik:")
    
    # DİKKAT KİRVE: Extra Trees feature importance sıralamana göre buradaki isimleri ve st.progress yüzdelerini değiştirmelisin!
    st.write("1. CT Toplam Zırh (CT Armor)")
    st.progress(0.85)
    st.write("2. Kalan Süre (Time Left)")
    st.progress(0.72)
    st.write("3. T Toplam Silah Değeri (T Equipment Value)")
    st.progress(0.68)
    st.write("4. CT Kask Sayısı (CT Helmets)")
    st.progress(0.55)
    st.write("5. T Hayatta Kalan Oyuncu (T Players Alive)")
    st.progress(0.49)

# --- FARK METRİKLERİ ---
st.markdown("---")
st.markdown("### ⚡ Anlık Üstünlük Durumu (CT - T Farkları)")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Skor Farkı",   f"{st.session_state['ct_score'] - st.session_state['t_score']:+d}")
m2.metric("HP Farkı",     f"{st.session_state['ct_health'] - st.session_state['t_health']:+d}")
m3.metric("Zırh Farkı",  f"{st.session_state['ct_armor'] - st.session_state['t_armor']:+d}")
m4.metric("Kask Farkı",  f"{st.session_state['ct_helmets'] - st.session_state['t_helmets']:+d}")
m5.metric("Para Farkı",  f"${st.session_state['ct_money'] - st.session_state['t_money']:+,}")
m6.metric("Oyuncu Farkı",f"{st.session_state['ct_alive'] - st.session_state['t_alive']:+d}")

# --- TAHMİN BÖLÜMÜ ---
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔥 ROUND SONUCUNU TAHMİN ET VE ANALİZİ BAŞLAT"):
    # --- VERİ TUTARLILIK KONTROLÜ (DATA VALIDATION) ---
    uyarilar = []
    if st.session_state['ct_alive'] == 0 and st.session_state['ct_health'] > 0:
        uyarilar.append("⚠️ CT takımı ölü görünüyor ama HP değerleri 0'dan büyük!")
    if st.session_state['t_alive'] == 0 and st.session_state['t_health'] > 0:
        uyarilar.append("⚠️ T takımı ölü görünüyor ama HP değerleri 0'dan büyük!")
    if st.session_state['ct_money'] > (st.session_state['ct_alive'] * 16000):
        uyarilar.append("⚠️ CT parası oyundaki maksimum limiti ($16.000 x Oyuncu) aşıyor!")
    if st.session_state['ct_health'] > (st.session_state['ct_alive'] * 100):
        uyarilar.append("⚠️ CT HP değeri mantıksız. Bir oyuncunun maksimum 100 HP'si olabilir.")
        
    if uyarilar:
        for uyari in uyarilar:
            st.warning(uyari)
        st.toast("Girdiğiniz bazı veriler CS2 oyun mekanikleriyle çelişiyor olabilir!", icon="🚨")
    # --------------------------------------------------

    # Tüm verileri modelin beklediği formata sokuyoruz
    row = {
        "time_left": st.session_state['time_left'],
        "ct_score": st.session_state['ct_score'], "t_score": st.session_state['t_score'],
        "ct_health": st.session_state['ct_health'], "t_health": st.session_state['t_health'],
        "ct_armor": st.session_state['ct_armor'], "t_armor": st.session_state['t_armor'],
        "ct_money": st.session_state['ct_money'], "t_money": st.session_state['t_money'],
        "ct_helmets": st.session_state['ct_helmets'], "t_helmets": st.session_state['t_helmets'],
        "ct_defuse_kits": st.session_state['ct_defuse_kits'],
        "ct_players_alive": st.session_state['ct_alive'], "t_players_alive": st.session_state['t_alive'],
        "map": selected_map,
        # Feature Engineering Farkları
        "score_diff": st.session_state['ct_score'] - st.session_state['t_score'],
        "health_diff": st.session_state['ct_health'] - st.session_state['t_health'],
        "money_diff": st.session_state['ct_money'] - st.session_state['t_money'],
        "player_diff": st.session_state['ct_alive'] - st.session_state['t_alive'],
        "armor_diff": st.session_state['ct_armor'] - st.session_state['t_armor'],
        "helmet_diff": st.session_state['ct_helmets'] - st.session_state['t_helmets'],
        "bomb_planted_int": int(bomb_planted),
    }

    # Silahları row sözlüğüne ekleme
    for w in weapons_ct_rifles + weapons_t_rifles + weapons_snipers + weapons_pistols + grenades + other_weapons:
        row[w] = st.session_state.get(w, 0)

    input_df = pd.DataFrame([row])

    # Eksik sütunları 0 ile doldurma (Model güvenliği için)
    if hasattr(pipeline, "feature_names_in_"):
        expected_columns = pipeline.feature_names_in_
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0.0
        input_df = input_df[expected_columns]
    else:
        # Eğer manuel listeye güveniyorsak
        missing_cols = ['t_weapon_galilar', 'ct_weapon_mac10', 't_grenade_hegrenade', 'ct_weapon_xm1014', 't_weapon_fiveseven', 't_grenade_flashbang', 'ct_weapon_ssg08', 'ct_weapon_ak47', 'ct_weapon_elite', 'ct_weapon_glock', 'ct_weapon_p250', 't_weapon_tec9', 'ct_grenade_hegrenade', 't_weapon_sawedoff', 'ct_grenade_smokegrenade', 'ct_weapon_ump45', 't_weapon_mac10', 'ct_grenade_molotovgrenade', 'ct_weapon_awp', 'ct_weapon_deagle', 'ct_grenade_decoygrenade', 'ct_grenade_incendiarygrenade', 't_weapon_famas', 't_grenade_decoygrenade', 't_weapon_m4a4', 't_weapon_mag7', 't_weapon_mp5sd', 'ct_weapon_m249', 'ct_weapon_m4a1s', 'ct_weapon_m4a4', 't_weapon_sg553', 'ct_weapon_p90', 't_weapon_mp7', 't_grenade_smokegrenade', 'ct_weapon_famas', 'ct_weapon_scar20', 't_weapon_glock', 't_weapon_awp', 't_weapon_aug', 't_weapon_p2000', 't_grenade_incendiarygrenade', 't_grenade_molotovgrenade', 't_weapon_r8revolver', 'ct_weapon_mp7', 't_weapon_g3sg1', 'ct_grenade_flashbang', 't_weapon_ak47', 't_weapon_cz75auto', 'ct_weapon_fiveseven', 'ct_weapon_usps', 't_weapon_usps', 't_weapon_xm1014', 't_weapon_m4a1s', 't_weapon_p250', 't_weapon_deagle', 'ct_weapon_aug', 't_weapon_ump45', 't_weapon_elite', 'ct_weapon_mp9', 'ct_weapon_galilar', 'ct_weapon_p2000', 't_weapon_mp9', 't_weapon_p90', 't_weapon_nova', 'ct_weapon_sg553', 'ct_weapon_mp5sd', 'ct_weapon_mag7', 't_weapon_bizon', 'ct_weapon_tec9', 'ct_weapon_cz75auto', 't_weapon_ssg08', 'ct_weapon_nova']
        for col in missing_cols:
            if col not in input_df.columns:
                input_df[col] = 0.0

    try:
        proba   = pipeline.predict_proba(input_df)[0]
        ct_prob = float(proba[1])
        t_prob  = 1 - ct_prob
        label   = "CT" if ct_prob >= 0.5 else "T"
        conf    = abs(ct_prob - 0.5) * 2  

        color  = "#60a5fa" if label == "CT" else "#f87171"
        emoji  = "👮‍♂️" if label == "CT" else "🥷"
        conf_txt = "🔥 Yüksek" if conf > 0.6 else "⚡ Orta" if conf > 0.3 else "🤔 Düşük"

        st.markdown(f"""
        <div style="background:{'rgba(96,165,250,0.1)' if label=='CT' else 'rgba(248,113,113,0.1)'};
          border:2px solid {color}; border-radius:14px; padding:30px; margin:20px 0;
          box-shadow:0 0 30px {color}44; text-align:center;">
          <div style="font-size:3.5rem; margin-bottom:10px;">{emoji}</div>
          <div style="font-family:'Rajdhani',sans-serif; font-size:1rem; color:{color}; font-weight:700; letter-spacing:0.1em;">YAPAY ZEKA TAHMİNİ</div>
          <div style="font-size:3rem; font-weight:900; color:{color}; margin:10px 0; font-family:'Rajdhani',sans-serif;">
            {label} KAZANIR
          </div>
          <div style="color:#e2e8f0; font-size:1.1rem; margin-top:15px; background: rgba(0,0,0,0.4); display:inline-block; padding: 10px 20px; border-radius: 8px;">
            CT Olasılığı: <strong style="color:#60a5fa;">{ct_prob*100:.1f}%</strong> &nbsp;|&nbsp;
            T Olasılığı: <strong style="color:#f87171;">{t_prob*100:.1f}%</strong> &nbsp;|&nbsp;
            Güven Seviyesi: <strong style="color:#f59e0b;">{conf_txt}</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_gauge, col_shap = st.columns([1, 1.8])

        with col_gauge:
            st.markdown("<h4 style='text-align: center; color: #f8fafc; font-family: Rajdhani;'>Dominasyon Metresi</h4>", unsafe_allow_html=True)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ct_prob * 100,
                number={"suffix": "%", "font": {"color": "#f8fafc", "size": 45, "family": "Rajdhani"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#94a3b8"}},
                    "bar":  {"color": color, "thickness": 0.4},
                    "bgcolor": "#1e293b",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50],  "color": "rgba(248,113,113,0.2)"},
                        {"range": [50, 100],"color": "rgba(96,165,250,0.2)"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": ct_prob * 100},
                },
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col_shap:
            st.markdown("<h4 style='text-align: center; color: #f8fafc; font-family: Rajdhani;'>🧠 Karar Mekanizması (SHAP Waterfall)</h4>", unsafe_allow_html=True)
            try:
                if hasattr(pipeline, 'named_steps'):
                    model_step_name = list(pipeline.named_steps.keys())[-1]
                    prep_step_name = list(pipeline.named_steps.keys())[0]
                    classifier = pipeline.named_steps[model_step_name]
                    preprocessor = pipeline.named_steps[prep_step_name]
                    transformed_data = preprocessor.transform(input_df)
                    try:
                        feature_names = preprocessor.get_feature_names_out()
                    except:
                        feature_names = input_df.columns
                else:
                    classifier = pipeline
                    transformed_data = input_df
                    feature_names = input_df.columns

                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer(transformed_data)
                
                if len(shap_values.shape) == 3:
                    shap_values_to_plot = shap_values[:, :, 1] 
                else:
                    shap_values_to_plot = shap_values

                shap_values_to_plot.feature_names = feature_names

                fig_shap = plt.figure(figsize=(9, 5))
                ax = fig_shap.gca()
                fig_shap.patch.set_facecolor('#0b0e14')
                ax.set_facecolor('#0b0e14')
                ax.xaxis.label.set_color('#9ca3af')
                ax.yaxis.label.set_color('#9ca3af')
                ax.tick_params(axis='x', colors='#f8fafc')
                ax.tick_params(axis='y', colors='#f8fafc')
                ax.spines['bottom'].set_color('#374151')
                ax.spines['top'].set_color('#0b0e14') 
                ax.spines['right'].set_color('#0b0e14')
                ax.spines['left'].set_color('#374151')
                
                shap.plots.waterfall(shap_values_to_plot[0], max_display=8, show=False)
                st.pyplot(fig_shap)
            except Exception as e:
                st.warning(f"SHAP analizinde uyumsuzluk. Hata: {e}")

        # --- OTURUM GEÇMİŞİNE EKLEME ---
        st.session_state['history'].append({
            "Harita": selected_map,
            "Süre": st.session_state['time_left'],
            "CT Skoru": st.session_state['ct_score'],
            "T Skoru": st.session_state['t_score'],
            "Tahmin": label,
            "Güven": f"%{max(ct_prob,t_prob)*100:.1f}",
        })

    except Exception as e:
        st.error(f"❌ Tahmin hatası: {e}")

# --- OTURUM GEÇMİŞİ ---
st.markdown("---")
st.markdown("### 📂 Oturum Tahmin Geçmişi")
if len(st.session_state.get('history', [])) > 0:
    history_df = pd.DataFrame(st.session_state['history'])
    st.dataframe(history_df, use_container_width=True)
else:
    st.info("Henüz bir tahmin yapılmadı. Analiz çalıştırıldığında geçmiş burada listelenecektir.")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem;">
  <strong>CRISP-DM Deployment Fazı</strong> | YBS3259 Makine Öğrenmesi Final Projesi | Streamlit & Extra Trees Entegrasyonu
</div>
""", unsafe_allow_html=True)