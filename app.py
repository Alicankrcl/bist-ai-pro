"""
BIST Pro Terminal & AI Trader
Mobil uyumlu, ML destekli, Groq tabanlı profesyonel finans terminali.
"""
import streamlit as st
import sqlite3
import yfinance as yf
import pandas as pd
import math
from datetime import datetime
from dotenv import load_dotenv
import os
from groq import Groq
from ml_engine import run_ml_prediction
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── SAYFA ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BIST Pro Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── ORTAM ──────────────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ALLOWED_USERS = {
    os.environ.get("USER1_ID", "Alican"): os.environ.get("USER1_PW", "1234"),
    os.environ.get("USER2_ID", "Halil"):  os.environ.get("USER2_PW", "1234"),
}
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY bulunamadi. .env dosyasini kontrol edin.")
    st.stop()

# ── GÜVENLİK ──────────────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None

if not st.session_state.authenticated:
    st.markdown("""
    <style>
    #MainMenu,footer{visibility:hidden}
    [data-testid="stSidebarCollapsedControl"]{visibility:visible !important; display:flex !important;}
    html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#060810!important;color:white}
    </style>""", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;margin-top:100px;color:#fff'>⚡ AI Quant Pro — Giriş</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:rgba(239,68,68,.12);border:1px solid #ef4444;border-radius:8px;
                padding:10px 14px;margin:12px auto;max-width:460px;text-align:center;
                font-size:11px;color:#fca5a5;'>
        ⚖️ <b>YASAL UYARI (SPK):</b> Bu platform tamamen deneysel ve eğitim amaçlıdır.
        Sunulan analizler <b>yatırım tavsiyesi (YTD) niteliği taşımaz.</b>
        Finansal kararlarınızın sorumluluğu tamamen size aittir.
    </div>""", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        user_input = st.text_input("Kullanıcı Adı")
        pwd_input  = st.text_input("Şifre", type="password")
        if st.button("Giriş Yap", use_container_width=True, type="primary"):
            if user_input in ALLOWED_USERS and ALLOWED_USERS[user_input] == pwd_input:
                st.session_state.authenticated = True
                st.session_state.username = user_input
                st.rerun()
            else:
                st.error("❌ Hatalı kullanıcı adı veya şifre!")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ── VERİTABANI ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_db():
    conn = sqlite3.connect("bist_agent.db", check_same_thread=False)
    try:
        conn.execute("SELECT ts FROM messages LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("DROP TABLE IF EXISTS messages")
        conn.commit()
    conn.execute("""CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT, role TEXT, content TEXT, ts DATETIME)""")
    conn.commit()
    return conn

conn = get_db()

def save_msg(role, content):
    uid = st.session_state.get("username", "s1")
    conn.execute("INSERT INTO messages(session_id,role,content,ts) VALUES(?,?,?,?)",
                 (uid, role, content, datetime.now()))
    conn.commit()

def load_msgs():
    uid = st.session_state.get("username", "s1")
    return conn.execute(
        "SELECT role,content FROM messages WHERE session_id=? ORDER BY id", (uid,)
    ).fetchall()

def clear_msgs():
    uid = st.session_state.get("username", "s1")
    conn.execute("DELETE FROM messages WHERE session_id=?", (uid,))
    conn.commit()

# ── BIST HAVUZ (Yıldız + Ana Pazar + Seçme Hisseler) ─────────────────────────
BIST_100 = [
    # BIST 100 (Yıldız Pazar)
    "AEFES","AGHOL","AHGAZ","AKBNK","AKCNS","AKFGY","AKFYE","AKSA","AKSEN","ALARK",
    "ALBRK","ALFAS","ARCLK","ASELS","ASTOR","BERA","BIENY","BIMAS","BIOEN","BOBET",
    "BRSAN","BRYAT","BUCIM","CANTE","CCOLA","CIMSA","CWENE","DOAS","DOHOL","ECILC",
    "EGEEN","EKGYO","ENERY","ENJSA","ENKAI","EREGL","EUPWR","EUREN","FROTO","GARAN",
    "GENIL","GESAN","GLYHO","GUBRF","GWIND","HALKB","HEKTS","IMASM","IPEKE","ISCTR",
    "ISGYO","ISMEN","IZENR","KALES","KARSN","KCAER","KCHOL","KLSER","KMPUR","KONTR",
    "KONYA","KOZAA","KOZAL","KRDMD","KZBGY","MAVI","MGROS","MIATK","ODAS","OTKAR",
    "OYAKC","PENTA","PETKM","PGSUS","PNLSN","QUAGR","REEDR","SAHOL","SASA","SDTTR",
    "SISE","SKBNK","SMRTG","SOKM","TABGD","TAVHL","TCELL","THYAO","TKFEN","TOASO",
    "TSKB","TTKOM","TTRAK","TUKAS","TUPRS","ULKER","VAKBN","VESBE","VESTL","YEOTK",
    "YKBNK","YYLGD","ZOREN",
    # Ana Pazar & Ek Hisseler
    "ACSEL","ADEL","ADESE","AGHOL","AGROT","AGYO","AHSGY","AKENR","AKMGY","AKSGY",
    "ALCTL","ALKIM","ALMAD","ANELE","ANSGR","ARASE","ARDYZ","ARENA","ARMDA","ARZUM",
    "ATAGY","ATEKS","ATLAS","ATSYH","AVHOL","AVOD","AVTUR","AYCES","AYDEM","AYEN",
    "BAGFS","BAKAB","BALAT","BANVT","BARMA","BASGZ","BAYRK","BFREN","BIGCH","BINHO",
    "BIOEN","BLCYT","BMSTL","BNTAS","BOSSA","BRISA","BRKVY","BRLNS","BRMEN","BTCIM",
    "BUCIM","BURCE","BURVA","CASA","CELHA","CEMAS","CEMTS","CEOEM","CIMSA","CLEBI",
    "CMBTN","CONSE","COSMO","CRDFA","CRFSA","CUSAN","DAGI","DAPGM","DARDL","DENGE",
    "DERHL","DERIM","DESA","DEVA","DGATE","DGGYO","DGNMO","DIRIT","DMSAS","DNISI",
    "DOBUR","DOCO","DOGUB","DURDO","DYOBY","DZGYO","ECILC","ECZYT","EDIP","EGEEN",
    "EGEPO","EGGUB","EGPRO","EGSER","EMKEL","EMNIS","ENERY","ENJSA","ENKAI","ENSRI",
    "EPLAS","ERSU","ESCAR","ESCOM","ESEN","ETILR","ETYAT","EUHOL","EUPWR","EUREN",
    "FADE","FENER","FLAP","FONET","FORMT","FORTE","FRIGO","GEDIK","GEDZA","GENIL",
    "GENTS","GEREL","GLYHO","GLRYH","GMTAS","GOKNR","GOLTS","GOODY","GOZDE","GRSEL",
    "GRTRK","GSDDE","GSDHO","GSRAY","GZNMI","HATEK","HTTBT","HUBVC","HUNER","HURGZ",
    "ICBCT","IDEA","IHEVA","IHGZT","IHLAS","IHLGM","IHYAY","INDES","INFO","INTEM",
    "INVEO","IPEKE","ISGSY","ISMEN","ISSEN","IZFAS","IZINV","JANTS","KAPLM","KARYE",
    "KATMR","KAYSE","KERVT","KFEIN","KGYO","KIMMR","KLMSN","KNFRT","KONKA","KOTON",
    "KRDMA","KRONT","KRSTL","KUTPO","KUYAS","KUVVT","LIDER","LILAK","LKMNH","LOGO",
    "LUKSK","MAALT","MACKO","MAGEN","MAKIM","MAKTK","MANAS","MARBL","MARKA","MARTI",
    "MAVI","MEDTR","MEGAP","MEKAG","MERCN","MERIT","MERKO","METRO","MHRGY","MIPAZ",
    "MMCAS","MNDRS","MNDTR","MOBTL","MPARK","MRGYO","MSGYO","MTRKS","MTRYO","MZHLD",
    "NATEN","NETAS","NIBAS","NTGAZ","NTHOL","NUGYO","NUHCM","OBAMS","OBASE","ODAS",
    "OFSYM","ONCSM","ORCAY","ORGE","OSMEN","OSTIM","OYLUM","OYAKC","OYYAT","OZGYO",
    "OZKGY","PAMEL","PAPIL","PARSN","PASEU","PCILT","PEKGY","PENGD","PENTA","PGSUS",
    "PINSU","PKART","PLTUR","PNLSN","POLHO","POLTK","PRDGS","PRKAB","PRKME","PSGYO",
    "QUAGR","RALYH","RAYSG","REEDR","RGYAS","RODRG","ROYAL","RUBNS","SAFKR","SAMAT",
    "SANEL","SANFM","SANKO","SARKY","SASA","SAYAS","SDTTR","SEGYO","SEKFK","SEKUR",
    "SELEC","SELGD","SELVA","SILVR","SNGYO","SNICA","SNKRN","SODSN","SRVGY","SUMAS",
    "SUNTK","SUWEN","TABGD","TATEN","TATGD","TAVHL","TERA","TETMT","TEZOL","TGSAS",
    "TLMAN","TMPOL","TNZTP","TOASO","TRGYO","TRILC","TSGYO","TSPOR","TUCLK","TUREX",
    "TURSG","UFUK","ULAS","ULUUN","UNLU","USAK","UZERB","VAKFN","VAKKO","VANGD",
    "VBTYZ","VERUS","VERTU","VESBE","VESTL","VKFYO","VKGYO","VRGYO","YAPRK","YATAS",
    "YEOTK","YGGYO","YGYO","YUNSA","YYLGD","ZEDUR","ZRGYO",
]
# Tekrarlı sembolleri kaldır, sıralı tut
BIST_ALL = sorted(set(BIST_100))

# ── VERİ ÇEKME ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=120, show_spinner=False)
def fetch(symbol: str) -> dict:
    try:
        sym = symbol.upper().replace(".IS", "")
        # yf.download ile Adj Close garantisi
        df_raw = yf.download(f"{sym}.IS", period="1y", progress=False)
        if df_raw.empty:
            return {"error": f"{sym} için veri bulunamadı"}

        # Sütun adları bazen MultiIndex gelebilir (Örn: yfinance v0.2.x sonrası yf.download)
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.droplevel(1)

        # Temettü ve bölünme düzeltilmiş kapanış
        if "Adj Close" in df_raw.columns:
            close = df_raw["Adj Close"]
        else:
            close = df_raw["Close"]

        last  = df_raw.iloc[-1]
        prev  = df_raw.iloc[-2] if len(df_raw) > 1 else last
        
        # Değişim yüzdesini güncel fiyata (Close) göre mi Adj Close'a göre mi yapmalı?
        # Günlük yüzde değişimi için Adj Close daha sağlıklıdır.
        chg   = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100
        price = float(df_raw["Close"].iloc[-1]) # Görünen fiyat her zaman son gerçek kapanıştır

        # ── TRADINGVIEW MATEMATİĞİ (WILDER'S RMA) ────────────────────────────
        def rma(series, length):
            """Wilder's Smoothing (RMA)"""
            return series.ewm(alpha=1/length, adjust=False).mean()

        # 1. TRADINGVIEW UYUMLU RSI(14)
        delta = close.diff()
        gain  = delta.where(delta > 0, 0.0)
        loss  = -delta.where(delta < 0, 0.0)
        avg_gain = rma(gain, 14)
        avg_loss = rma(loss, 14)
        rs = avg_gain / avg_loss
        rsi = float(100 - (100 / (1 + rs)).iloc[-1])

        # 2. TRADINGVIEW UYUMLU MACD (12, 26, 9)
        e1   = close.ewm(span=12, adjust=False).mean()
        e2   = close.ewm(span=26, adjust=False).mean()
        macd = e1 - e2
        sig  = macd.ewm(span=9, adjust=False).mean()

        # 3. EMA
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        trend = "Yükseliş" if ema20 > ema50 else "Düşüş"

        # 4. Bollinger (SMA bazlı)
        sma20  = close.rolling(20).mean()
        std20  = close.rolling(20).std()
        upper_bb = float((sma20 + 2 * std20).iloc[-1])
        lower_bb = float((sma20 - 2 * std20).iloc[-1])

        # 5. TRADINGVIEW UYUMLU ATR(14)
        tr1 = df_raw["High"] - df_raw["Low"]
        tr2 = (df_raw["High"] - close.shift()).abs()
        tr3 = (df_raw["Low"]  - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = float(rma(tr, 14).iloc[-1])

        # 6. Destek / Direnç
        support    = float(df_raw["Low"].tail(20).min())
        resistance = float(df_raw["High"].tail(20).max())

        # Hacim
        vol_ma20   = float(df_raw["Volume"].rolling(20).mean().iloc[-1])
        vol_curr   = float(last["Volume"])
        vol_surge  = vol_curr > vol_ma20 * 1.5

        # İstatistiksel fiyat bandı (1 ay)
        rets      = close.pct_change().dropna()
        daily_vol = rets.std()
        monthly_v = daily_vol * math.sqrt(22)
        pred_high = price * (1 + monthly_v * 1.96)
        pred_low  = price * (1 - monthly_v * 1.96)

        tk = yf.Ticker(f"{sym}.IS")
        info = tk.info
        fk   = info.get("trailingPE")
        pddd = info.get("priceToBook")

        return {
            "symbol":     sym,
            "price":      round(price, 2),
            "chg":        round(chg, 2),
            "rsi":        round(rsi, 2),
            "macd":       round(float(macd.iloc[-1]), 2),
            "macd_sig":   round(float(sig.iloc[-1]), 2),
            "ema20":      round(ema20, 2),
            "ema50":      round(ema50, 2),
            "trend":      trend,
            "upper_bb":   round(upper_bb, 2),
            "lower_bb":   round(lower_bb, 2),
            "atr":        round(atr, 2),
            "support":    round(support, 2),
            "resistance": round(resistance, 2),
            "vol_surge":  vol_surge,
            "pred_high":  round(pred_high, 2),
            "pred_low":   round(pred_low, 2),
            "fk":         round(fk,   2) if isinstance(fk,   (int, float)) else None,
            "pddd":       round(pddd, 2) if isinstance(pddd, (int, float)) else None,
            "_df":        df_raw,     # ML için ham veri
        }
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def get_ml(symbol: str) -> dict:
    """ML tahminini önbellekte tutar (1 saat)"""
    tk = yf.Ticker(f"{symbol.upper()}.IS")
    df = tk.history(period="2y")
    if df.empty:
        return {"error": "Veri yok"}
    return run_ml_prediction(df)

# ── CSS (MODERN FİNANS TERMİNALİ) ────────────────────────────────────────────
st.markdown("""
<style>
/* ── GENEL ────────────────────────────────────────────────────── */
#MainMenu,footer{visibility:hidden}
[data-testid="stSidebarCollapsedControl"]{
    visibility:visible !important;
    display:flex !important;
    z-index:99999 !important;
}
header[data-testid="stHeader"]{
    background:transparent !important;
    color:#e2e8f0 !important;
}

html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{
    background:#060810 !important;
    color:#e2e8f0;
    font-family:'Inter',-apple-system,sans-serif;
}
[data-testid="stSidebar"]{
    background:#0b0f1a !important;
    border-right:1px solid #1a2035;
}

/* ── MOBİL SIDEBAR FIX ───────────────────────────────────────── */
@media(max-width:768px){
    [data-testid="stSidebar"]{
        position:fixed!important;
        z-index:999!important;
        height:100vh!important;
        top:0!important;
        left:0!important;
        width:85vw!important;
        max-width:320px!important;
        transform:translateX(0);
        transition:transform .25s ease;
    }
    [data-testid="stSidebar"][aria-expanded="false"]{
        transform:translateX(-100%)!important;
    }
    /* Mobilde içerik taşmaları */
    [data-testid="stMain"]{
        padding:0 8px!important;
        overflow-x:hidden!important;
    }
    .metric-row{
        flex-wrap:wrap!important;
    }
    .kart{
        width:calc(50% - 8px)!important;
        min-width:140px!important;
    }
}

/* Sabit mobil menü toggle butonu */
.mobile-toggle{
    display:none;
}
@media(max-width:768px){
    .mobile-toggle{
        display:flex!important;
        position:fixed;
        top:12px;left:12px;
        z-index:1100;
        background:#1a2035;
        border:1px solid #2a3555;
        border-radius:8px;
        padding:6px 10px;
        cursor:pointer;
        align-items:center;
        gap:6px;
        font-size:13px;
        color:#94a3b8;
        box-shadow:0 4px 20px rgba(0,0,0,.5);
    }
}

/* ── PREMIUM KARTLAR ─────────────────────────────────────────── */
.kart{
    background:rgba(255,255,255,.03);
    border:1px solid #1a2035;
    border-radius:12px;
    padding:16px 18px;
    margin-bottom:12px;
    position:relative;
    overflow:hidden;
    transition:border-color .2s, box-shadow .2s, transform .15s;
    flex:1 1 160px;
}
.kart::before{
    content:'';
    position:absolute;
    inset:0;
    background:linear-gradient(135deg,rgba(59,130,246,.06) 0%,transparent 60%);
    pointer-events:none;
}
.kart:hover{
    border-color:#3b82f6;
    box-shadow:0 0 18px rgba(59,130,246,.18), 0 4px 20px rgba(0,0,0,.3);
    transform:translateY(-2px);
}
.kart.green-glow:hover{ border-color:#10b981; box-shadow:0 0 18px rgba(16,185,129,.18); }
.kart.red-glow:hover  { border-color:#ef4444; box-shadow:0 0 18px rgba(239,68,68,.18);  }

.kart-label{
    font-size:10px;
    color:#4b5a7b;
    letter-spacing:1.2px;
    text-transform:uppercase;
    font-weight:600;
    margin-bottom:8px;
}
.kart-val{
    font-size:24px;
    font-weight:700;
    color:#ffffff;
    line-height:1.1;
}
.kart-val.green { color:#10b981 }
.kart-val.red   { color:#ef4444 }
.kart-val.blue  { color:#3b82f6 }
.kart-val.amber { color:#f59e0b }

/* ── ML GÜVEN KARTI ──────────────────────────────────────────── */
.ml-card{
    background:linear-gradient(135deg,rgba(139,92,246,.08) 0%,rgba(59,130,246,.06) 100%);
    border:1px solid rgba(139,92,246,.3);
    border-radius:12px;
    padding:16px 20px;
    margin:12px 0;
    position:relative;
    overflow:hidden;
}
.ml-card::before{
    content:'';
    position:absolute;
    top:-40px;right:-40px;
    width:100px;height:100px;
    background:radial-gradient(circle,rgba(139,92,246,.15) 0%,transparent 70%);
}
.ml-title{font-size:11px;color:#8b5cf6;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:10px;font-weight:700}
.ml-prob{ font-size:36px;font-weight:800; }
.ml-badge{
    display:inline-block;
    padding:4px 12px;
    border-radius:20px;
    font-size:11px;
    font-weight:700;
    letter-spacing:.5px;
    margin-top:6px;
}
.badge-buy      { background:rgba(16,185,129,.15);  color:#10b981; border:1px solid rgba(16,185,129,.3);  }
.badge-strongbuy{ background:rgba(16,185,129,.25);  color:#34d399; border:1px solid rgba(16,185,129,.5);  }
.badge-sell     { background:rgba(239,68,68,.15);   color:#ef4444; border:1px solid rgba(239,68,68,.3);   }
.badge-strgsell { background:rgba(239,68,68,.25);   color:#f87171; border:1px solid rgba(239,68,68,.5);   }
.badge-neutral  { background:rgba(100,116,139,.15); color:#94a3b8; border:1px solid rgba(100,116,139,.3); }

/* ── YTD BAR ─────────────────────────────────────────────────── */
.ytd-bar{
    background:rgba(239,68,68,.07);
    border-left:3px solid #ef4444;
    border-radius:4px;
    padding:8px 14px;
    font-size:11px;
    color:#94a3b8;
    margin-bottom:18px;
    line-height:1.6;
}

/* ── HEADER ──────────────────────────────────────────────────── */
.app-header{
    display:flex;
    align-items:center;
    gap:14px;
    padding-bottom:14px;
    border-bottom:1px solid #1a2035;
    margin-bottom:16px;
}
.app-title{
    font-size:22px;
    font-weight:800;
    background:linear-gradient(90deg,#3b82f6 0%,#8b5cf6 100%);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    margin:0;
}
.live-badge{
    background:rgba(16,185,129,.12);
    border:1px solid rgba(16,185,129,.3);
    color:#10b981;
    font-size:10px;
    font-weight:700;
    padding:3px 10px;
    border-radius:20px;
    letter-spacing:1px;
    animation:pulse 2s infinite;
}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}

/* ── CHAT ────────────────────────────────────────────────────── */
[data-testid="stChatMessage"]{
    background:rgba(255,255,255,.025) !important;
    border:1px solid #1a2035 !important;
    border-radius:10px;
    backdrop-filter:blur(8px);
}
[data-testid="stChatInput"] textarea{
    background:rgba(255,255,255,.04) !important;
    border:1px solid #1a2035 !important;
    border-radius:20px !important;
    color:#fff !important;
}
[data-testid="stChatInput"] textarea:focus{
    border-color:#3b82f6 !important;
    box-shadow:0 0 0 2px rgba(59,130,246,.2) !important;
}

/* ── TABS ────────────────────────────────────────────────────── */
[data-testid="stTabs"] button{color:#4b5a7b!important;font-weight:600;font-size:12px;letter-spacing:.5px}
[data-testid="stTabs"] button[aria-selected="true"]{color:#fff!important;border-bottom-color:#3b82f6!important}

/* ── SCROLLBAR ───────────────────────────────────────────────── */
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#1a2035;border-radius:4px}

/* ── METRIC ROW ──────────────────────────────────────────────── */
.metric-row{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px}

hr{border-color:#1a2035!important}
</style>
""", unsafe_allow_html=True)

# ── SOL PANEL ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <style>
    /* SOL PANEL ÖZEL CSS */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b0f1a 0%, #05070d 100%) !important;
        border-right: 1px solid rgba(59, 130, 246, 0.12) !important;
    }
    
    /* Input & Selectbox Styling */
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stTextInput input {
        background: rgba(15, 23, 42, 0.4) !important;
        border: 1px solid rgba(59, 130, 246, 0.15) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div:hover,
    [data-testid="stSidebar"] .stTextInput input:hover {
        border-color: rgba(59, 130, 246, 0.4) !important;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.1) !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div:focus-within,
    [data-testid="stSidebar"] .stTextInput input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.25) !important;
    }
    
    /* Buton Styling */
    [data-testid="stSidebar"] button {
        background: rgba(30, 41, 59, 0.3) !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        border-radius: 8px !important;
        color: #94a3b8 !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebar"] button:hover {
        background: rgba(59, 130, 246, 0.1) !important;
        border-color: rgba(59, 130, 246, 0.4) !important;
        color: #60a5fa !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(59, 130, 246, 0.1) !important;
        margin: 1.5rem 0 !important;
    }
    </style>
    
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px">
        <div style="background:linear-gradient(135deg,#3b82f6,#8b5cf6);width:36px;height:36px;
                    border-radius:10px;display:flex;align-items:center;justify-content:center;
                    font-size:18px;flex-shrink:0;box-shadow:0 0 15px rgba(59,130,246,0.3)">⚡</div>
        <div>
            <div style="font-size:16px;font-weight:800;color:#fff;letter-spacing:0.5px">AI Quant Pro</div>
            <div style="font-size:11px;color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:1px">BIST Terminal</div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.caption(f"👤 **Aktif Oturum:** {st.session_state.username}")
    st.divider()

    st.markdown('<div style="font-size:10px;color:#64748b;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;font-weight:600">Enstrüman Seçimi</div>', unsafe_allow_html=True)
    liste_sec = st.selectbox("BIST 100", BIST_100, index=3, label_visibility="collapsed")
    ozel      = st.text_input("Özel Sembol", placeholder="🔍 Sembol Arama (Örn: TATEN)", label_visibility="collapsed")
    selected  = ozel.upper().strip() if ozel else liste_sec

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧹 Temizle", use_container_width=True):
            clear_msgs()
            uid = st.session_state.get("username", "s1") + "_mentor"
            conn.execute("DELETE FROM messages WHERE session_id=?", (uid,))
            conn.commit()
            st.session_state.pop("messages", None)
            st.session_state.pop("mentor_messages", None)
            st.rerun()
    with c2:
        if st.button("🚪 Çıkış", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()

# ── BAŞLIK & YTD ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1 class="app-title">BIST Pro Terminal & AI Trader</h1>
    <span class="live-badge">● LIVE</span>
</div>
<div class="ytd-bar">
    ⚖️ <b>YASAL UYARI (SPK Mevzuatı):</b> Burada yer alan yapay zeka analizleri, makine öğrenmesi
    tahminleri ve algoritmik veriler tamamen eğitim amaçlıdır.
    <b>Kesinlikle yatırım danışmanlığı (YTD) kapsamında değildir.</b>
    Finansal kararlarınızın sorumluluğu tamamen size aittir.
</div>
""", unsafe_allow_html=True)

# ── SEKMELER ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["⚡  DAY TRADING & AI", "🔍  BIST TARAYICI", "📚  BORSA AKADEMİ & MENTOR"])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — DAY TRADING & AI
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    with st.spinner(f"{selected} verileri yükleniyor..."):
        data = fetch(selected)

    if "error" in data:
        st.error(f"Veri hatası veya bağlantı problemi: {data['error']}")
        st.info("Lütfen sol panelden farklı bir hisse senedi seçiniz veya birazdan tekrar deneyiniz.")
    else:

        sign     = "+" if data["chg"] >= 0 else ""
        chg_cls  = "green" if data["chg"] >= 0 else "red"
        tr_cls   = "green" if data["trend"] == "Yükseliş" else "red"
        rsi_cls  = "red"   if data["rsi"] > 70 else ("green" if data["rsi"] < 35 else "amber")
        vol_txt  = "Yüksek Hacim" if data["vol_surge"] else "Normal Hacim"
        vol_cls  = "amber" if data["vol_surge"] else ""

        # ── ÜST METRİK KARTLARI ──────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="kart {'green-glow' if data['chg']>=0 else 'red-glow'}">
                <div class="kart-label">Anlık Fiyat</div>
                <div class="kart-val">{data['price']} ₺</div>
                <div class="kart-val" style="font-size:14px" class="{chg_cls}">{sign}{data['chg']}%</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kart {'green-glow' if tr_cls=='green' else 'red-glow'}">
                <div class="kart-label">EMA Trend</div>
                <div class="kart-val {tr_cls}">{data['trend'].upper()}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kart">
                <div class="kart-label">Destek / Direnç (20G)</div>
                <div class="kart-val" style="font-size:17px">
                    <span style="color:#10b981">{data['support']} ₺</span>
                    <span style="color:#4b5a7b;font-size:13px;padding:0 4px">/</span>
                    <span style="color:#ef4444">{data['resistance']} ₺</span>
                </div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="kart">
                <div class="kart-label">Hacim Durumu</div>
                <div class="kart-val {vol_cls}">{vol_txt}</div>
            </div>""", unsafe_allow_html=True)

        # ── DETAY KARTLARI (RSI, MACD, Bollinger, ATR, Fiyat Bandı) ─────────────
        c5, c6, c7, c8, c9 = st.columns(5)
        with c5:
            st.markdown(f"""
            <div class="kart">
                <div class="kart-label">RSI (14)</div>
                <div class="kart-val {rsi_cls}">{data['rsi']}</div>
            </div>""", unsafe_allow_html=True)
        with c6:
            macd_cls = "green" if data["macd"] > data["macd_sig"] else "red"
            st.markdown(f"""
            <div class="kart">
                <div class="kart-label">MACD / Sinyal</div>
                <div class="kart-val {macd_cls}" style="font-size:15px">{data['macd']} / {data['macd_sig']}</div>
            </div>""", unsafe_allow_html=True)
        with c7:
            st.markdown(f"""
            <div class="kart">
                <div class="kart-label">Bollinger Alt / Üst</div>
                <div class="kart-val" style="font-size:15px">{data['lower_bb']} / {data['upper_bb']}</div>
            </div>""", unsafe_allow_html=True)
        with c8:
            st.markdown(f"""
            <div class="kart">
                <div class="kart-label">ATR (14) — Stop Ref.</div>
                <div class="kart-val blue">{data['atr']} ₺</div>
            </div>""", unsafe_allow_html=True)
        with c9:
            st.markdown(f"""
            <div class="kart">
                <div class="kart-label">1 Aylık Vol. Bandı</div>
                <div class="kart-val" style="font-size:15px">
                    <span style="color:#10b981">{data['pred_low']} ₺</span>
                    <span style="color:#4b5a7b;padding:0 4px">—</span>
                    <span style="color:#ef4444">{data['pred_high']} ₺</span>
                </div>
            </div>""", unsafe_allow_html=True)

        # ── ML TAHMİN KARTI ───────────────────────────────────────────────────────
        st.markdown("---")
        with st.spinner("🧠 ML Modeli eğitiliyor ve tahmin üretiliyor..."):
            ml = get_ml(selected)

        if ml.get("error"):
            st.warning(f"ML Modeli: {ml['error']}")
        else:
            sig_map = {
                "GÜÇLÜ ALIŞ":  ("badge-strongbuy", "🟢 GÜÇLÜ ALIŞ"),
                "ALIŞ":        ("badge-buy",        "🟢 ALIŞ"),
                "NÖTR":        ("badge-neutral",    "⚪ NÖTR"),
                "SATIŞ":       ("badge-sell",       "🔴 SATIŞ"),
                "GÜÇLÜ SATIŞ": ("badge-strgsell",   "🔴 GÜÇLÜ SATIŞ"),
            }
            badge_cls, badge_txt = sig_map.get(ml["signal"], ("badge-neutral", ml["signal"]))
            prob_color = "#10b981" if ml["prob_up"] >= 55 else ("#ef4444" if ml["prob_up"] <= 45 else "#f59e0b")

            ml_col1, ml_col2 = st.columns([2, 3])
            with ml_col1:
                st.markdown(f"""
                <div class="ml-card">
                    <div class="ml-title">🧠 ML TAHMİN MODELİ (RandomForest {ml['n_trees']} ağaç)</div>
                    <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
                        <div>
                            <div style="font-size:10px;color:#8b5cf6;margin-bottom:4px">YÜKSELİŞ OLASILĞI</div>
                            <div class="ml-prob" style="color:{prob_color}">{ml['prob_up']}%</div>
                        </div>
                        <div>
                            <div style="font-size:10px;color:#4b5a7b;margin-bottom:4px">SINYAL</div>
                            <span class="ml-badge {badge_cls}">{badge_txt}</span>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
            with ml_col2:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Model Güven Skoru", f"{ml['confidence']}%",
                              help="Modelin en yüksek sınıfa verdiği olasılık")
                    st.metric("CV Doğruluğu (Yaklaşık)", f"{ml['accuracy']}%",
                              help="Zaman serisi çapraz doğrulama ortalama doğruluğu")
                with col_b:
                    st.metric("Düşüş Olasılığı", f"{ml['prob_down']}%")
                    st.metric("Eğitim Periyodu", "2 yıl geçmiş veri")

        # ── AI TRADER CHAT ────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div style="font-size:14px;font-weight:700;color:#cbd5e1;margin-bottom:10px">🤖 AI Pro Trader Asistanı (Groq GPT-OSS 120B)</div>', unsafe_allow_html=True)

        # Chat kutusunu genişleten CSS
        st.markdown("""
        <style>
        /* Chat mesajları tam genişliğe yayılsın, ferah padding */
        .stChatMessage {
            background: rgba(15, 23, 42, 0.4);
            border: 1px solid rgba(51, 65, 85, 0.5);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
        }
        .stChatMessage [data-testid="chatAvatarIcon-user"] {
            background-color: #3b82f6;
        }
        .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
            background-color: #8b5cf6;
        }
        </style>
        """, unsafe_allow_html=True)

        if "messages" not in st.session_state:
            st.session_state.messages = []
            for role, content in load_msgs():
                st.session_state.messages.append({"role": role, "content": content})

        # Fixed height (scroll) kaldırıldı, sayfa yüksekliğine göre doğal uzayacak
        chat_box = st.container()
        with chat_box:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

        if prompt := st.chat_input(f"{selected} için strateji, analiz veya soru sorun..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            save_msg("user", prompt)

            with chat_box:
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("model"):
                    with st.spinner("AI analiz yapıyor..."):
                        try:
                            ml_ctx = ""
                            if not ml.get("error"):
                                ml_ctx = (
                                    f"\n🧠 ML MODEL ÇIKTISI (RandomForest, 200 ağaç, 2 yıl veri):\n"
                                    f"  • Yükseliş Olasılığı : %{ml['prob_up']}\n"
                                    f"  • Düşüş Olasılığı   : %{ml['prob_down']}\n"
                                    f"  • Sinyal Etiketi     : {ml['signal']}\n"
                                    f"  • Model Güven Skoru  : %{ml['confidence']}\n"
                                    f"  • CV Doğruluğu       : %{ml['accuracy']}\n"
                                )

                            ctx = (
                                f"━━━ GERÇEK ZAMANLI VERİ PAKETİ ━━━\n"
                                f"Hisse     : {selected}\n"
                                f"Fiyat     : {data['price']} TL  |  Günlük Değ.: {sign}{data['chg']}%\n"
                                f"Destek    : {data['support']} TL  |  Direnç : {data['resistance']} TL\n"
                                f"ATR (14)  : {data['atr']} TL  (Stop-Loss referansı)\n"
                                f"RSI (14)  : {data['rsi']}\n"
                                f"MACD      : {data['macd']}  |  Sinyal : {data['macd_sig']}\n"
                                f"Bollinger : Alt {data['lower_bb']} TL  |  Üst {data['upper_bb']} TL\n"
                                f"EMA Trend : {data['trend']} (EMA20 vs EMA50)\n"
                                f"Vol. Bandı: {data['pred_low']} – {data['pred_high']} TL  (1 aylık %95 istatistiksel bant)\n"
                                f"Hacim     : {'Yüksek — Normalin 1.5x üzeri' if data['vol_surge'] else 'Normal'}\n"
                                f"{ml_ctx}"
                                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            )

                            sys_prompt = (
                                "### KİMLİK & GÖREV\n"
                                "Sen Borsa İstanbul (BIST) uzmanı, Wall Street ve hedge-fund deneyimli üst düzey bir "
                                "Quant Analist ve Algoritmik Trader yapay zekasısın. "
                                "Görevin: Sana iletilen GERÇEK ZAMANLI teknik veriler ve ML model çıktısına dayanarak "
                                "SOMUT, CESUR ve MATEMATİKSEL analizler üretmek.\n\n"
                                "### MUTLAK KURALLAR (İhlal edilemez)\n"
                                "1. 'Yatırım tavsiyesi veremem', 'fiyat tahmini yapamam', 'bu bir öneri değildir' "
                                "gibi KAÇAMAÇ ifadeleri ASLA kullanma. Bu davranış yasaktır.\n"
                                "2. Kullanıcı fiyat hedefi, tahmin veya senaryo istediğinde elimdeki verilere "
                                "dayanarak NET rakamlar, bantlar ve olasılıklar üret. 'Belki', 'kesin değil' "
                                "yerine 'veriler X'i işaret ediyor, Y senaryosu ağırlıklı' tarzında konuş.\n"
                                "3. Tüm analizlerin matematiksel temeli mevcut: RSI, MACD, ATR, Bollinger, "
                                "volatilite bandı ve 200-ağaçlı RandomForest ML skoru. Bu verileri gerekçe olarak kullan.\n"
                                "4. Her yanıtın EN SONUNA (başına değil) tek satır halinde yasal şerhi ekle — "
                                "başka yerde sorumluluk reddi yazma, analizi engelleme.\n"
                                "5. Teknik analizi (RSI, MACD, ATR, Destek/Direnç) yaptıktan sonra mutlaka "
                                "**'Halk Dili / Yeni Başlayan Özeti'** adında ayrı bir bölüm aç. Burada teknik "
                                "terimleri adeta masada oturan bir arkadaşa anlatır gibi, 'Yani kısacası kral, bu "
                                "hisse şurada sıkışmış, alıcılar baskın gelirse patlar' tarzında son derece net, "
                                "samimi ve anlaşılır bir dille özetle.\n\n"
                                "### VERİ PAKETİ\n"
                                f"{ctx}\n"
                                "### ZORUNLU YANIT FORMATI\n\n"
                                "## 📊 Profesyonel Quant Analizi\n"
                                "RSI, MACD crossover durumu, Bollinger pozisyonu, Hacim anomalisi ve "
                                "ML modelinin güven skorunu birleştirerek matematiksel ve profesyonel yorum yap.\n\n"
                                "## 🎯 Fiyat Senaryoları & Trade Planı\n"
                                "**Baz Senaryo:** 1-5 günlük beklenen hareket bandını TL cinsinden ver.\n"
                                "**Alım Bölgesi (Entry):** Destek seviyesi + ATR baz alınarak en mantıklı giriş noktası.\n"
                                "**Hedef Fiyat (Take Profit):** Direnç ve Bollinger Üst Bandı baz alınarak net TL rakamı.\n"
                                "**Zarar Kes (Stop-Loss):** `Fiyat - ATR` formülüyle hesapla, kesin TL rakamı ver.\n"
                                "**Risk/Ödül Oranı:** Hesapla ve yaz.\n\n"
                                "## ☕ Halk Dili / Yeni Başlayan Özeti\n"
                                "Teknik terimleri bırak, borsa bilmeyen bir arkadaşına masada durumu anlatır gibi "
                                "samimi ve net 2-3 cümle. (Örn: 'Kısacası kral, hisse dirençte zorlanıyor...')\n\n"
                                "---\n"
                                "⚠️ *YTD: Bu analiz yapay zeka algoritması tarafından eğitim amaçlı üretilmiştir. "
                                "Finansal kararlarınızın sorumluluğu tamamen size aittir.*"
                            )

                            groq_msgs = [{"role": "system", "content": sys_prompt}]
                            for m in st.session_state.messages[-8:]:
                                role = "assistant" if m["role"] == "model" else "user"
                                groq_msgs.append({"role": role, "content": m["content"]})

                            resp = client.chat.completions.create(
                                model="openai/gpt-oss-120b",
                                messages=groq_msgs,
                                temperature=0.15,   # Daha deterministik ve tutarlı yanıtlar
                                max_tokens=1500,    # Daha uzun, daha detaylı analizler
                            )
                            reply = resp.choices[0].message.content
                            st.markdown(reply)
                            st.session_state.messages.append({"role": "model", "content": reply})
                            save_msg("model", reply)

                        except Exception as e:
                            st.error(f"Hata: {e}")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — BIST QUANT FIRSAT TARAYICI
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    import streamlit.components.v1 as components

    # ── TARAYICI BAŞLIK ───────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">
        <div style="font-size:16px;font-weight:800;color:#e2e8f0;letter-spacing:0.3px">
            Quant Fırsat Tarayıcı
        </div>
        <span style="background:rgba(139,92,246,.12);border:1px solid rgba(139,92,246,.3);
                     color:#a78bfa;font-size:9px;font-weight:700;padding:2px 8px;
                     border-radius:12px;letter-spacing:0.8px">PARALLEL ENGINE</span>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"RSI · Momentum · PD/DD · F/K · Trend · Hacim — 10 üzerinden algoritmik puanlama  |  Havuz: {len(BIST_ALL)} hisse  |  24 saat önbellek")

    # ── TEK HİSSE TARAMA (thread-safe) ───────────────────────────────────────
    def _scan_single(sym: str) -> dict | None:
        d = fetch(sym)
        if "error" in d:
            return None
        score = 0.0
        pv = d["pddd"]
        if isinstance(pv, (int, float)):
            if   pv < 1.0: score += 2.0
            elif pv < 1.5: score += 1.5
            elif pv < 2.5: score += 1.0
            elif pv < 4.0: score += 0.5
        fk = d["fk"]
        if isinstance(fk, (int, float)) and fk > 0:
            if   fk < 8:   score += 1.5
            elif fk < 12:  score += 1.0
            elif fk < 18:  score += 0.5
        rv = d["rsi"]
        if isinstance(rv, (int, float)):
            if   rv < 30:  score += 2.0
            elif rv < 35:  score += 1.5
            elif rv < 40:  score += 1.0
            elif rv < 45:  score += 0.5
        if d["macd"] > d["macd_sig"]:
            score += 1.0
            if d["macd"] > 0:
                score += 0.5
        if d["trend"] == "Yükseliş":
            score += 1.5
        if d["vol_surge"]:
            score += 1.0
        bb_range = d["upper_bb"] - d["lower_bb"]
        if bb_range > 0:
            bb_pos = (d["price"] - d["lower_bb"]) / bb_range
            if bb_pos < 0.2:
                score += 0.5
        score = min(round(score, 1), 10.0)
        if   score >= 7.5: sinyal = "GÜÇLÜ FIRSAT"
        elif score >= 5.5: sinyal = "FIRSAT"
        elif score >= 3.5: sinyal = "NÖTR"
        elif score >= 2.0: sinyal = "ZAYIF"
        else:              sinyal = "KAÇIN"
        return {
            "Hisse": sym, "Fiyat": d["price"], "Değişim %": d["chg"],
            "RSI": d["rsi"], "MACD": d["macd"], "PD/DD": pv, "F/K": fk,
            "Trend": d["trend"],
            "Hacim": "Yüksek" if d["vol_surge"] else "Normal",
            "Puan": score, "Sinyal": sinyal,
        }

    # ── GÜNLÜK CACHE FONKSİYONU ──────────────────────────────────────────────
    @st.cache_data(ttl=86400, show_spinner=False)
    def run_full_scan(symbol_list: tuple) -> pd.DataFrame:
        """Tüm BIST havuzunu paralel tarar, 24 saat önbellekte tutar."""
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_map = {executor.submit(_scan_single, s): s for s in symbol_list}
            for future in as_completed(future_map):
                try:
                    r = future.result()
                    if r:
                        results.append(r)
                except Exception:
                    pass
        if not results:
            return pd.DataFrame()
        return (
            pd.DataFrame(results)
            .sort_values("Puan", ascending=False)
            .reset_index(drop=True)
        )

    # ── TARAMA BUTONU ────────────────────────────────────────────────────────
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        do_scan = st.button("Tüm BIST'i Tara", use_container_width=True, type="primary")
    with col_info:
        st.markdown(f'<div style="font-size:11px;color:#64748b;padding-top:8px">Paralel motor: {len(BIST_ALL)} hisse · 24 saat cache · ~30-60 saniye (ilk tarama)</div>', unsafe_allow_html=True)

    if do_scan:
        st.cache_data.clear()
        st.session_state.pop("screener_df", None)

    # ── TARAMAYI ÇALIŞTIR (cache varsa anında gelir) ─────────────────────────
    if do_scan or "screener_df" not in st.session_state:
        with st.spinner(f"{len(BIST_ALL)} hisse paralel taranıyor..."):
            df_scan = run_full_scan(tuple(BIST_ALL))
        if not df_scan.empty:
            st.session_state.screener_df = df_scan

    # ── SONUÇ GÖSTERME ───────────────────────────────────────────────────────
    df = st.session_state.get("screener_df", pd.DataFrame())

    if not df.empty:
        # ── KPI ÖZET KARTLARI ────────────────────────────────────────────────
        total_scanned = len(df)
        avg_chg       = round(df["Değişim %"].mean(), 2)
        bullish_count = len(df[df["Trend"] == "Yükseliş"])
        bearish_count = total_scanned - bullish_count
        top_opps      = len(df[df["Puan"] >= 5.5])
        avg_rsi       = round(df["RSI"].mean(), 1)

        mkt_color = "#10b981" if avg_chg >= 0 else "#ef4444"
        mkt_text  = "Pozitif" if avg_chg >= 0 else "Negatif"

        st.markdown(f"""
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin:16px 0">
            <div class="kart" style="flex:1;min-width:130px">
                <div class="kart-label">Taranan Hisse</div>
                <div class="kart-val blue">{total_scanned}</div>
            </div>
            <div class="kart {'green-glow' if avg_chg>=0 else 'red-glow'}" style="flex:1;min-width:130px">
                <div class="kart-label">Ort. Değişim</div>
                <div class="kart-val" style="color:{mkt_color}">{'+'if avg_chg>=0 else ''}{avg_chg}%</div>
                <div style="font-size:10px;color:#64748b;margin-top:4px">Piyasa: {mkt_text}</div>
            </div>
            <div class="kart" style="flex:1;min-width:130px">
                <div class="kart-label">Yükseliş / Düşüş</div>
                <div class="kart-val" style="font-size:17px">
                    <span style="color:#10b981">{bullish_count}</span>
                    <span style="color:#334155;padding:0 4px">/</span>
                    <span style="color:#ef4444">{bearish_count}</span>
                </div>
            </div>
            <div class="kart green-glow" style="flex:1;min-width:130px">
                <div class="kart-label">Fırsat (≥5.5)</div>
                <div class="kart-val green">{top_opps}</div>
            </div>
            <div class="kart" style="flex:1;min-width:130px">
                <div class="kart-label">Ort. RSI</div>
                <div class="kart-val amber">{avg_rsi}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── İNTERAKTİF FİLTRE BARI ──────────────────────────────────────────
        st.markdown('<div style="font-size:10px;color:#64748b;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;font-weight:600">Filtre & Sıralama</div>', unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns([1, 1, 1])
        with fc1:
            min_puan = st.slider("Min. Puan", 0.0, 10.0, 0.0, 0.5, key="scr_puan")
        with fc2:
            sinyal_filtre = st.multiselect(
                "Sinyal", ["GÜÇLÜ FIRSAT", "FIRSAT", "NÖTR", "ZAYIF", "KAÇIN"],
                default=["GÜÇLÜ FIRSAT", "FIRSAT", "NÖTR", "ZAYIF", "KAÇIN"],
                key="scr_sinyal"
            )
        with fc3:
            sort_col = st.selectbox("Sırala", ["Puan", "Değişim %", "RSI", "PD/DD", "F/K", "Fiyat"], key="scr_sort")

        filtered = df[
            (df["Puan"] >= min_puan) &
            (df["Sinyal"].isin(sinyal_filtre))
        ].sort_values(sort_col, ascending=(sort_col in ["RSI", "PD/DD", "F/K"]))

        st.caption(f"Filtrelenmiş: {len(filtered)} / {total_scanned} hisse")

        # ── HTML TABLO (st.components.v1.html ile render — asla bozulmaz) ────
        def _badge_html(s):
            cm = {
                "GÜÇLÜ FIRSAT": ("#10b981", "rgba(16,185,129,.15)", "rgba(16,185,129,.4)"),
                "FIRSAT":       ("#60a5fa", "rgba(59,130,246,.12)", "rgba(59,130,246,.3)"),
                "NÖTR":         ("#94a3b8", "rgba(100,116,139,.12)", "rgba(100,116,139,.3)"),
                "ZAYIF":        ("#fbbf24", "rgba(251,191,36,.12)", "rgba(251,191,36,.3)"),
                "KAÇIN":        ("#ef4444", "rgba(239,68,68,.12)", "rgba(239,68,68,.3)"),
            }
            fg, bg, bd = cm.get(s, cm["NÖTR"])
            return f'<span style="background:{bg};color:{fg};border:1px solid {bd};padding:2px 10px;border-radius:12px;font-size:10px;font-weight:700;letter-spacing:.3px">{s}</span>'

        def _chg(v):
            if v > 0:  return f'<span style="color:#10b981;font-weight:600">+{v}%</span>'
            if v < 0:  return f'<span style="color:#ef4444;font-weight:600">{v}%</span>'
            return f'<span style="color:#94a3b8">{v}%</span>'

        def _rsi(v):
            if v > 70:  return f'<span style="color:#ef4444;font-weight:600">{v}</span>'
            if v < 35:  return f'<span style="color:#10b981;font-weight:600">{v}</span>'
            return f'<span style="color:#f59e0b">{v}</span>'

        def _bar(v):
            pct = min(v / 10 * 100, 100)
            c = "#10b981" if v >= 7.5 else ("#3b82f6" if v >= 5.5 else ("#f59e0b" if v >= 3.5 else "#ef4444"))
            return (f'<div style="display:flex;align-items:center;gap:6px">'
                    f'<span style="font-weight:700;font-size:12px;color:{c};min-width:26px">{v}</span>'
                    f'<div style="flex:1;height:5px;background:rgba(255,255,255,.06);border-radius:3px;overflow:hidden">'
                    f'<div style="width:{pct}%;height:100%;background:{c};border-radius:3px"></div>'
                    f'</div></div>')

        def _trend(t):
            if t == "Yükseliş": return '<span style="color:#10b981">▲ Yükseliş</span>'
            return '<span style="color:#ef4444">▼ Düşüş</span>'

        def _v(x):
            return "—" if x is None else str(x)

        # Satırlar
        rows = ""
        for _, r in filtered.iterrows():
            rows += (
                f'<tr>'
                f'<td style="font-weight:700;color:#e2e8f0;font-size:12px">{r["Hisse"]}</td>'
                f'<td style="color:#cbd5e1">{r["Fiyat"]} ₺</td>'
                f'<td>{_chg(r["Değişim %"])}</td>'
                f'<td>{_rsi(r["RSI"])}</td>'
                f'<td style="color:#cbd5e1">{_v(r["PD/DD"])}</td>'
                f'<td style="color:#cbd5e1">{_v(r["F/K"])}</td>'
                f'<td>{_trend(r["Trend"])}</td>'
                f'<td style="font-size:11px;color:#94a3b8">{r["Hacim"]}</td>'
                f'<td style="min-width:90px">{_bar(r["Puan"])}</td>'
                f'<td>{_badge_html(r["Sinyal"])}</td>'
                f'</tr>'
            )

        th_style = "text-align:left;padding:10px 8px;color:#64748b;font-size:10px;letter-spacing:.8px;text-transform:uppercase;font-weight:600;border-bottom:1px solid #1a2035"

        full_html = f"""
        <html>
        <head>
        <style>
            * {{ margin:0; padding:0; box-sizing:border-box; }}
            body {{ background:transparent; font-family:'Inter',-apple-system,sans-serif; color:#e2e8f0; }}
            table {{ width:100%; border-collapse:separate; border-spacing:0; font-size:12px; }}
            thead th {{ position:sticky; top:0; background:#0b0f1a; z-index:2; }}
            tbody tr {{ border-bottom:1px solid rgba(26,32,53,.5); transition:background .15s; }}
            tbody tr:hover {{ background:rgba(59,130,246,.08); }}
            tbody td {{ padding:9px 8px; vertical-align:middle; }}
        </style>
        </head>
        <body>
        <div style="overflow-x:auto;max-height:600px;overflow-y:auto">
        <table>
            <thead>
                <tr>
                    <th style="{th_style}">Hisse</th>
                    <th style="{th_style}">Fiyat</th>
                    <th style="{th_style}">Değişim</th>
                    <th style="{th_style}">RSI</th>
                    <th style="{th_style}">PD/DD</th>
                    <th style="{th_style}">F/K</th>
                    <th style="{th_style}">Trend</th>
                    <th style="{th_style}">Hacim</th>
                    <th style="{th_style};min-width:100px">Puan</th>
                    <th style="{th_style}">Sinyal</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        </div>
        </body>
        </html>
        """

        # Yüksekliği satır sayısına göre ayarla (min 200, max 650)
        table_height = min(max(len(filtered) * 42 + 50, 200), 650)
        components.html(full_html, height=table_height, scrolling=True)

        # ── ALT YTD ─────────────────────────────────────────────────────────
        st.markdown("""
        <div style="margin-top:14px;font-size:10px;color:#475569;line-height:1.6">
            ⚠️ <b>YTD:</b> Puanlama: PD/DD&lt;1→+2, &lt;1.5→+1.5, &lt;2.5→+1, &lt;4→+0.5 |
            F/K&lt;8→+1.5, &lt;12→+1, &lt;18→+0.5 | RSI&lt;30→+2, &lt;35→+1.5, &lt;40→+1, &lt;45→+0.5 |
            MACD&gt;Sinyal→+1, pozitif→+0.5 | Yükseliş Trendi→+1.5 | Yüksek Hacim→+1 |
            Bollinger alt bant→+0.5. Bu veriler yatırım tavsiyesi niteliği taşımaz.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#334155">
            <div style="font-size:48px;margin-bottom:12px;opacity:0.3">🔍</div>
            <div style="font-size:14px;font-weight:600;color:#64748b">Henüz tarama yapılmadı</div>
            <div style="font-size:12px;color:#475569;margin-top:6px">
                Butona tıklayarak tüm BIST hisselerini paralel olarak tarayın.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════════
    #  TAB 3 — BORSA AKADEMİ & MENTOR
    # ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">
        <div style="font-size:16px;font-weight:800;color:#e2e8f0;letter-spacing:0.3px">
            Borsa Akademi & AI Mentor
        </div>
        <span style="background:rgba(236,72,153,.12);border:1px solid rgba(236,72,153,.3);
                     color:#f472b6;font-size:9px;font-weight:700;padding:2px 8px;
                     border-radius:12px;letter-spacing:0.8px">LIVE TUTOR</span>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Borsa terminolojisini öğrenin, stratejiler kurun ve yapay zeka ile kendinizi geliştirin.")

    # ── SEVİYE SEÇİMİ ─────────────────────────────────────────────────────────
    st.markdown('<div style="margin-top:16px;font-size:12px;color:#cbd5e1;font-weight:600;margin-bottom:8px">Eğitim Seviyenizi Seçin:</div>', unsafe_allow_html=True)
    mentor_level = st.radio(
        "Seviye Seçimi",
        ["🌱 Çırak (Halk Dili)", "📈 Orta Seviye", "🧠 Pro Quant"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # ── DİNAMİK BAĞLAM (CANLI VERİ) ───────────────────────────────────────────
    live_context = ""
    try:
        if 'selected' in locals() and selected:
            d = fetch(selected)
            if "error" not in d:
                live_context = (
                    f"\n\n[SİSTEM BİLGİSİ: Kullanıcı şu an sol panelde {selected} hissesini seçmiş durumda. "
                    f"Anlık Fiyat: {d.get('price', 'Bilinmiyor')}, RSI: {d.get('rsi', 'Bilinmiyor')}, "
                    f"Trend: {d.get('trend', 'Bilinmiyor')}. Bir konuyu anlatırken veya örnek verirken "
                    f"mutlaka bu hissenin anlık verilerini kullanarak canlı örnekler ver.]"
                )
    except Exception:
        pass

    # ── MENTOR VERİTABANI İŞLEMLERİ ───────────────────────────────────────────
    def save_mentor_msg(role, content):
        uid = st.session_state.get("username", "s1") + "_mentor"
        conn.execute("INSERT INTO messages(session_id,role,content,ts) VALUES(?,?,?,?)",
                     (uid, role, content, datetime.now()))
        conn.commit()

    def load_mentor_msgs():
        uid = st.session_state.get("username", "s1") + "_mentor"
        return conn.execute("SELECT role,content FROM messages WHERE session_id=? ORDER BY id", (uid,)).fetchall()

    if "mentor_messages" not in st.session_state:
        st.session_state.mentor_messages = []
        for role, content in load_mentor_msgs():
            st.session_state.mentor_messages.append({"role": role, "content": content})

    # ── SPLIT-SCREEN EĞİTİM TASARIMI ──────────────────────────────────────────
    st.markdown("---")
    akademi_col1, akademi_col2 = st.columns([4, 6])

    with akademi_col1:
        st.markdown('<div style="font-size:14px;font-weight:700;color:#cbd5e1;margin-bottom:10px">📚 Müfredat & Rehber</div>', unsafe_allow_html=True)
        if "Çırak" in mentor_level:
            with st.expander("🤔 Hisse Senedi Nedir?", expanded=True):
                st.write("Hisse senedi, bir şirketin anaparasına ortak olduğunuzu gösteren resmi bir belgedir. Şirket büyürse sizin payınızın değeri de artar.")
            with st.expander("🧱 Destek ve Direnç Mantığı"):
                st.write("**Destek:** Fiyatın düşerken 'buradan aşağı zor iner' dediğimiz alıcıların beklediği zemin kat.\n\n**Direnç:** Fiyatın yükselirken 'burayı zor geçer' dediğimiz satıcıların beklediği tavan.")
            with st.expander("📈 RSI Nedir? (En Basit Hali)"):
                st.write("RSI, hissenin ne kadar 'şiştiğini' veya 'söndüğünü' gösterir. 70 üstü 'çok pahalandı, dikkat et', 30 altı 'çok ucuzladı, fırsat olabilir' demektir.")
        elif "Orta" in mentor_level:
            with st.expander("📊 MACD Nasıl Okunur?", expanded=True):
                st.write("MACD, iki farklı hareketli ortalamanın birbirine yakınlaşıp uzaklaşmasını ölçer. Mavi çizgi turuncuyu yukarı keserse 'AL', aşağı keserse 'SAT' sinyali olarak yorumlanır.")
            with st.expander("⚠️ RSI Uyumsuzlukları (Divergence)"):
                st.write("Fiyat yeni tepeler yaparken RSI daha düşük tepeler yapıyorsa (Negatif Uyumsuzluk), trendin gücü bitiyor demektir ve dönüş yakın olabilir.")
            with st.expander("🕯️ Mum Çubukları (Price Action)"):
                st.write("Pinbar, Engulfing (Yutan Mum) gibi formasyonlar alıcıların veya satıcıların o periyotta kimin galip geldiğini gösterir.")
        else:
            with st.expander("🧮 Sharpe Oranı (Sharpe Ratio)", expanded=True):
                st.write("Sharpe oranı, aldığınız ekstra risk başına ne kadar aşırı getiri (excess return) elde ettiğinizi ölçer. Oran 1.0 üzerindeyse strateji risk/getiri açısından başarılı kabul edilir.")
            with st.expander("🤖 Algoritmik Backtest Prensipleri"):
                st.write("Out-of-sample data, Look-ahead bias ve Survivorship bias algoritmik sistem testlerinde en sık yapılan hatalardır. Her model mutlaka Walk-Forward analizi ile test edilmelidir.")
            with st.expander("📉 Maksimum Düşüş (Max Drawdown)"):
                st.write("Portföyün zirveden dibe yaşadığı en büyük oransal kayıptır. Algoritmik sistemlerde Calmar Oranı (Yıllık Getiri / Max DD) ile risk iştahı belirlenir.")

    with akademi_col2:
        st.markdown('<div style="font-size:14px;font-weight:700;color:#cbd5e1;margin-bottom:10px">💬 AI Mentor Sohbeti</div>', unsafe_allow_html=True)
        mentor_chat_box = st.container()
        with mentor_chat_box:
            for m in st.session_state.mentor_messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

        if mentor_prompt := st.chat_input("Akademi Mentor'a bir konu veya strateji sor..."):
            st.session_state.mentor_messages.append({"role": "user", "content": mentor_prompt})
            save_mentor_msg("user", mentor_prompt)

            with mentor_chat_box:
                with st.chat_message("user"):
                    st.markdown(mentor_prompt)
                with st.chat_message("model"):
                    with st.spinner("Mentor yanıtlıyor..."):
                        try:
                            base_prompt = "Sen Borsa İstanbul yatırımcıları için özel bir yapay zeka eğitmenisin. (AI Mentor)"
                            if "Çırak" in mentor_level:
                                sys_level = (
                                    "Kullanıcı borsaya yeni başlamış bir çırak. Her şeyi sıfırdan, son derece basit, "
                                    "samimi ve sokak ağzıyla / halk diliyle anlat. Karmaşık finansal jargon kullanma. "
                                    "Bol bol benzetme (analoji) yap. 'Kardeşim, kral, dostum' gibi samimi bir üslup kullan."
                                )
                            elif "Orta" in mentor_level:
                                sys_level = (
                                    "Kullanıcı orta seviye bir yatırımcı. Temel kavramları biliyor, ancak strateji kurmayı öğrenmek istiyor. "
                                    "Öğretici, dengeli ve analitik bir dil kullan. Konseptleri teknik terimlerle ancak anlaşılır şekilde açıkla."
                                )
                            else:
                                sys_level = (
                                    "Kullanıcı profesyonel bir Quant ve algoritmik trader. Cevapların tamamen matematiksel formüllere, "
                                    "istatistiksel varyanslara, makine öğrenmesi yaklaşımlarına ve ileri düzey kantitatif analizlere dayanmalı. "
                                    "Kesinlikle basit anlatım yapma; akademik ve kurumsal bir dil kullan."
                                )

                            mentor_sys_prompt = f"{base_prompt}\n\n{sys_level}{live_context}"

                            m_msgs = [{"role": "system", "content": mentor_sys_prompt}]
                            for m in st.session_state.mentor_messages[-6:]:
                                role = "assistant" if m["role"] == "model" else "user"
                                m_msgs.append({"role": role, "content": m["content"]})

                            r_mentor = client.chat.completions.create(
                                model="openai/gpt-oss-120b",
                                messages=m_msgs,
                                temperature=0.3,
                                max_tokens=1000,
                            )
                            reply_mentor = r_mentor.choices[0].message.content
                            st.markdown(reply_mentor)
                            st.session_state.mentor_messages.append({"role": "model", "content": reply_mentor})
                            save_mentor_msg("model", reply_mentor)

                        except Exception as e:
                            st.error(f"Hata: {e}")
