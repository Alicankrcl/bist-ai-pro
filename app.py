import streamlit as st
import sqlite3
import yfinance as yf
import pandas as pd
import math
from datetime import datetime
from dotenv import load_dotenv
import os
from groq import Groq

# ── SAYFA ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Quant Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── ORTAM ──────────────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

ALLOWED_USERS = {
    os.environ.get("USER1_ID", "Alican"): os.environ.get("USER1_PW", "1234"),
    os.environ.get("USER2_ID", "Halil"): os.environ.get("USER2_PW", "1234")
}

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY bulunamadi. .env dosyasini kontrol edin.")
    st.stop()

# ── GÜVENLİK (ÇOKLU KULLANICI) ─────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None

if not st.session_state.authenticated:
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #0f1117 !important; color: white;}
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; margin-top: 120px;'>🔒 AI Quant Pro Güvenli Giriş</h2>", unsafe_allow_html=True)
    st.markdown("""
        <div style='background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; border-radius: 8px; padding: 12px; margin: 15px auto; max-width: 500px; text-align: center; font-size: 11px; color: #fca5a5;'>
            ⚠️ <b>ÖNEMLİ YASAL UYARI (SPK MEVZUATI):</b> Bu platformda sunulan içerik, grafik ve yapay zeka analizleri <b>kesinlikle yatırım tavsiyesi (YTD) niteliği taşımaz.</b> Tamamen deneysel ve eğitim amaçlı algoritmik bir çalışmadır.
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        user_input = st.text_input("Kullanıcı Adı:")
        pwd_input = st.text_input("Şifre:", type="password")
        if st.button("Giriş Yap", use_container_width=True, type="primary"):
            if user_input in ALLOWED_USERS and ALLOWED_USERS[user_input] == pwd_input:
                st.session_state.authenticated = True
                st.session_state.username = user_input
                st.rerun()
            else:
                st.error("❌ Hatalı kullanıcı adı veya şifre!")
    st.stop() # Doğru giriş yapılana kadar alttaki hiçbir kod çalışmaz
client = Groq(api_key=GROQ_API_KEY)

# ── VERİTABANI ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_db():
    conn = sqlite3.connect("bist_agent.db", check_same_thread=False)
    try:
        conn.execute("SELECT ts FROM messages LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("DROP TABLE IF EXISTS messages")
        conn.commit()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            ts DATETIME
        )
    """)
    conn.commit()
    return conn

conn = get_db()

def save_msg(role: str, content: str):
    user_id = st.session_state.get("username", "s1")
    conn.execute(
        "INSERT INTO messages (session_id, role, content, ts) VALUES (?,?,?,?)",
        (user_id, role, content, datetime.now())
    )
    conn.commit()

def load_msgs():
    user_id = st.session_state.get("username", "s1")
    return conn.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id",
        (user_id,)
    ).fetchall()

def clear_msgs():
    user_id = st.session_state.get("username", "s1")
    conn.execute("DELETE FROM messages WHERE session_id=?", (user_id,))
    conn.commit()

# ── BIST LİSTESİ (BIST 100 Öntanımlı) ──────────────────────────────────────────
BIST_100 = [
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
    "YKBNK","YYLGD","ZOREN"
]

# ── VERİ ÇEKME & TEKNİK ANALİZ İNDİKATÖRLERİ ───────────────────────────────────
@st.cache_data(ttl=120, show_spinner=False) # Day trading icin cache süresi 2 dakikaya indirildi
def fetch(symbol: str) -> dict:
    try:
        sym = symbol.upper().replace(".IS", "")
        tk  = yf.Ticker(f"{sym}.IS")
        df  = tk.history(period="1y")
        if df.empty:
            return {"error": f"{sym} icin veri bulunamadi"}

        close = df["Close"]
        last  = df.iloc[-1]
        prev  = df.iloc[-2] if len(df) > 1 else last
        chg   = ((last["Close"] - prev["Close"]) / prev["Close"]) * 100
        current_price = float(last["Close"])

        # RSI(14)
        delta = close.diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi   = float((100 - 100 / (1 + gain / loss)).iloc[-1])

        # MACD(12,26,9)
        e1   = close.ewm(span=12, adjust=False).mean()
        e2   = close.ewm(span=26, adjust=False).mean()
        macd = e1 - e2
        sig  = macd.ewm(span=9, adjust=False).mean()

        # EMA & Bollinger Bands (20)
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        trend = "Yukselis" if ema20 > ema50 else "Dusus"
        
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper_bb = float((sma20 + (std20 * 2)).iloc[-1])
        lower_bb = float((sma20 - (std20 * 2)).iloc[-1])

        # ATR (14) - Ortalama Gerçek Aralık (Day trading Stop-Loss için)
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = float(true_range.rolling(14).mean().iloc[-1])

        # Destek / Direnç (Son 20 günün dip ve tepesi)
        support = float(df["Low"].tail(20).min())
        resistance = float(df["High"].tail(20).max())

        # Hacim Analizi
        vol_sma20 = float(df["Volume"].rolling(20).mean().iloc[-1])
        vol_current = float(last["Volume"])
        vol_surge = "Yuksek" if vol_current > (vol_sma20 * 1.5) else "Normal"

        info = tk.info
        fk   = info.get("trailingPE")
        pddd = info.get("priceToBook")

        return {
            "symbol":   sym,
            "price":    round(current_price, 2),
            "chg":      round(chg, 2),
            "rsi":      round(rsi, 2),
            "macd":     round(float(macd.iloc[-1]), 2),
            "macd_sig": round(float(sig.iloc[-1]), 2),
            "ema20":    round(ema20, 2),
            "ema50":    round(ema50, 2),
            "trend":    trend,
            "upper_bb": round(upper_bb, 2),
            "lower_bb": round(lower_bb, 2),
            "atr":      round(atr, 2),
            "support":  round(support, 2),
            "resistance": round(resistance, 2),
            "vol_surge": vol_surge,
            "fk":       round(fk,   2) if isinstance(fk,   (int, float)) else None,
            "pddd":     round(pddd, 2) if isinstance(pddd, (int, float)) else None,
        }
    except Exception as e:
        return {"error": str(e)}

# ── CSS (MODERN UYGULAMA GÖRÜNÜMÜ) ─────────────────────────────────────────────
st.markdown("""
<style>
/* Streamlit Geliştirici Arayüzünü Tamamen Gizle */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Temel Uygulama Arka Planı (Şık, Mat, Koyu) */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #0f1117 !important;
    color: #e2e8f0;
    font-family: 'Inter', -apple-system, sans-serif;
}

[data-testid="stSidebar"] {
    background: #151821 !important;
    border-right: 1px solid #1f232e;
}

/* Premium Kartlar */
.kart {
    background: #1c202a;
    border: 1px solid #282d3b;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: all 0.2s ease;
}
.kart:hover {
    border-color: #3b82f6;
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.1);
}
.kart-label { font-size: 11px; color: #94a3b8; font-weight: 500; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px; }
.kart-val   { font-size: 26px; font-weight: 700; color: #ffffff; display: flex; align-items: baseline; gap: 8px;}
.kart-sub   { font-size: 14px; font-weight: 500; }
.green { color: #10b981 !important; }
.red   { color: #ef4444 !important; }
.blue  { color: #3b82f6 !important; }

/* Marka / Header */
.app-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 15px;
    border-bottom: 1px solid #1f232e;
    margin-bottom: 20px;
}
.app-header h1 {
    margin: 0;
    font-size: 24px;
    font-weight: 700;
    background: -webkit-linear-gradient(0deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-badge {
    background: rgba(59, 130, 246, 0.1);
    color: #3b82f6;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid rgba(59, 130, 246, 0.2);
}

/* Chat Arayüzü İyileştirmesi */
[data-testid="stChatMessage"] {
    background: #151821 !important;
    border: 1px solid #1f232e !important;
    border-radius: 12px;
    padding: 15px;
}
[data-testid="stChatInput"] {
    padding-bottom: 20px;
}
[data-testid="stChatInput"] textarea {
    background: #1c202a !important;
    border: 1px solid #282d3b !important;
    border-radius: 20px !important;
    color: #ffffff !important;
    padding: 10px 15px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #3b82f6 !important;
}

/* Tablar */
[data-testid="stTabs"] button { color: #94a3b8 !important; font-weight: 500; }
[data-testid="stTabs"] button[aria-selected="true"] { 
    color: #ffffff !important; 
    border-bottom-color: #3b82f6 !important; 
}
hr { border-color: #1f232e !important; }
</style>
""", unsafe_allow_html=True)

# ── SOL KENAR ÇUBUĞU ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 20px;">
            <div style="background: #3b82f6; color: white; width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 18px;">⚡</div>
            <h2 style="margin: 0; font-size: 18px; color: white;">AI Quant Pro</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.caption(f"👤 Aktif Kullanıcı: **{st.session_state.username}**")
    st.divider()

    st.markdown('<div style="color: #94a3b8; font-size: 12px; font-weight: 600; margin-bottom: 8px; text-transform: uppercase;">Aktif Hisse Seçimi</div>', unsafe_allow_html=True)
    liste_secimi = st.selectbox("BIST 100", BIST_100, index=3, label_visibility="collapsed") 
    
    st.markdown('<div style="color: #94a3b8; font-size: 12px; font-weight: 600; margin-top: 15px; margin-bottom: 8px; text-transform: uppercase;">Manuel Arama</div>', unsafe_allow_html=True)
    ozel_hisse = st.text_input("Özel Kod", placeholder="Örn: TATEN", label_visibility="collapsed")
    
    selected = ozel_hisse.upper().strip() if ozel_hisse else liste_secimi

    st.divider()
    st.markdown('<div style="color: #94a3b8; font-size: 12px; font-weight: 600; margin-bottom: 12px; text-transform: uppercase;">Bağlantılar</div>', unsafe_allow_html=True)
    st.markdown("🟢 BIST Veri Akışı Aktif (Day-Trading Uyumlu)")
    st.markdown("🟢 AI Trade Asistanı Hazır")

    st.divider()
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("Sohbeti Temizle", use_container_width=True):
            clear_msgs()
            st.session_state.pop("messages", None)
            st.rerun()
    with c_btn2:
        if st.button("Çıkış Yap", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()

# ── BAŞLIK ─────────────────────────────────────────────────────────────────────
st.markdown("""
    <div class="app-header">
        <h1>BIST Pro Terminal & AI Trader</h1>
        <span class="app-badge">LIVE ALGORITHM</span>
    </div>
    <div style="background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; border-radius: 4px; padding: 10px 14px; margin-bottom: 20px; font-size: 11px; color: #f87171; line-height: 1.5;">
        ⚖️ <b>YASAL UYARI (SPK Duyurusu):</b> Burada yer alan yatırım bilgi, yorum ve tavsiyeleri <b>yatırım danışmanlığı kapsamında değildir.</b> Yatırım danışmanlığı hizmeti; aracı kurumlar, portföy yönetim şirketleri, mevduat kabul etmeyen bankalar ile müşteri arasında imzalanacak yatırım danışmanlığı sözleşmesi çerçevesinde sunulmaktadır. Burada yer alan yapay zeka modelleri ve algoritmalar tamamen deneysel olup, finansal kararlarınızın sorumluluğu tamamen kendinize aittir. <b>(YTD)</b>
    </div>
""", unsafe_allow_html=True)

# ── SEKMELER ───────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["⚡ DAY TRADING & AI ANALİZ", "🔍 BIST 100 FIRSAT TARAYICI"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TRADING & AI
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    with st.spinner(f"{selected} verileri analiz ediliyor..."):
        data = fetch(selected)

    if "error" in data:
        st.error(f"Veri hatasi: {data['error']}")
    else:
        sign    = "+" if data["chg"] >= 0 else ""
        chg_cls = "green" if data["chg"] >= 0 else "red"
        trend_cls = "green" if data["trend"] == "Yukselis" else "red"
        vol_cls = "blue" if data["vol_surge"] == "Yuksek" else ""

        # Gelişmiş Trading Kartları
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown(f"""
            <div class="kart">
                <div class="kart-label">Anlık Fiyat & Değişim</div>
                <div class="kart-val">{data["price"]} ₺ <span class="kart-sub {chg_cls}">{sign}{data["chg"]}%</span></div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kart">
                <div class="kart-label">Kısa Vade Trend</div>
                <div class="kart-val {trend_cls}">{data["trend"].upper()}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kart">
                <div class="kart-label">Destek / Direnç (20G)</div>
                <div class="kart-val" style="font-size: 18px;">{data["support"]} ₺ / {data["resistance"]} ₺</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="kart">
                <div class="kart-label">Hacim Durumu (20G Ort)</div>
                <div class="kart-val {vol_cls}">{data["vol_surge"].upper()}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── AI TRADER CHAT ──
    st.markdown('<h3 style="font-size: 16px; margin-top: 20px; color: #cbd5e1;">🤖 AI Pro Trader Asistanı</h3>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
        for role, content in load_msgs():
            st.session_state.messages.append({"role": role, "content": content})

    chat_box = st.container(height=450)
    with chat_box:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    if prompt := st.chat_input(f"{selected} için alım-satım stratejisi veya analiz isteyin..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_msg("user", prompt)

        with chat_box:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("model"):
                with st.spinner("AI Trading Algoritması Çalışıyor..."):
                    try:
                        ctx = ""
                        if "error" not in data:
                            ctx = (
                                f"HISSE: {selected}\n"
                                f"Fiyat: {data['price']} TL (Degisim: {sign}{data['chg']}%)\n"
                                f"Destek: {data['support']} TL | Direnc: {data['resistance']} TL\n"
                                f"ATR (14) - Ortalama Gercek Aralik (Stop Loss belirlemek icin): {data['atr']} TL\n"
                                f"Bollinger Bantlari (20,2): Alt {data['lower_bb']} TL | Ust {data['upper_bb']} TL\n"
                                f"RSI(14): {data['rsi']}\n"
                                f"MACD: {data['macd']} (Sinyal: {data['macd_sig']})\n"
                                f"Trend (EMA 20/50): {data['trend']}\n"
                                f"Hacim Durumu: {data['vol_surge']}\n"
                            )

                        sys_prompt = (
                            "Sen kurumsal firmalara danismanlik veren, Wall Street kalitesinde ust duzey bir Quant ve Day Trader Yapay Zekasin.\n"
                            "Kullanici public bir yatirimci. Profesyonel, net, güven veren ve doğrudan hedefe yonelik bir dil kullan.\n\n"
                            "Sana Python tarafindan gercek zamanli teknik ve algoritmik veriler verildi:\n"
                            f"{ctx}\n\n"
                            "YANIT FORMATI (Bu basliklari birebir kullan):\n\n"
                            "## 📌 Piyasa Özeti (Anlaşılır Dilde)\n"
                            "Borsa jargonuna bogmadan, hissenin su anki psikolojisini ve alim-satim baskisini 2 cumle ile acikla. "
                            "Ornek: 'Hisse su an guclu bir alim dalgasinda ancak direnc seviyesine cok yakin, biraz yorulmus gorunuyor.'\n\n"
                            "## 🎯 Anlık & Günlük Trade Stratejisi\n"
                            "Verilen ATR, Bollinger ve Destek/Direnc verilerini kullanarak:\n"
                            "- **Mantikli Alim (Entry) Bolgesi:** Nereden alinabilir?\n"
                            "- **Hedef Fiyat (Take Profit):** Direnc veya Bollinger Ust Bandina gore nereden satilabilir?\n"
                            "- **Zarar Kes (Stop-Loss):** Fiyattan ATR degerini cikararak veya destek altina gore kesin bir stop-loss seviyesi (Rakam olarak) ver.\n\n"
                            "## 📊 Teknik Göstergelerin Durumu\n"
                            "RSI, MACD ve Hacmi kisaca (madde madde) yorumla. Ornegin 'Hacim normalin ustunde, bu yukselisi destekliyor' gibi.\n\n"
                            "Asla kesin kazanc garantisi verme. Yanitinin EN SONUNA kucuk puntolarla (Markdown caption: <br><br> *Yasal Uyari: Burada yer alan stratejiler yapay zeka tarafindan algoritmik olarak uretilmis olup, yatirim danismanligi kapsaminda degildir.*) ekle."
                        )

                        # Groq Mesaj Geçmişi Formatı
                        groq_messages = [{"role": "system", "content": sys_prompt}]
                        for m in st.session_state.messages[-6:]:
                            role = "assistant" if m["role"] == "model" else "user"
                            groq_messages.append({"role": role, "content": m["content"]})

                        response = client.chat.completions.create(
                            model="openai/gpt-oss-120b",
                            messages=groq_messages,
                            temperature=0.2,
                            max_tokens=1024
                        )
                        reply = response.choices[0].message.content
                        disclaimer_footer = (
                            "\n\n---\n"
                            "⚠️ **YASAL UYARI (YTD):** *Burada yer alan analiz ve algoritmik veriler tamamen eğitim amaçlı üretilmiş olup, "
                            "kesinlikle yatırım danışmanlığı kapsamında değildir. Gerçekleşecek işlemlerden doğacak zararlardan kullanıcı sorumludur.*"
                        )
                        reply_with_footer = reply + disclaimer_footer
                        st.markdown(reply_with_footer)
                        st.session_state.messages.append({"role": "model", "content": reply_with_footer})
                        save_msg("model", reply_with_footer)

                    except Exception as e:
                        st.error(f"Hata: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ISKONTOLU HISSELER (BIST 100 SCREENER)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<h3 style="font-size: 16px; margin-top: 10px; color: #cbd5e1;">BIST 100 Algoritmik Tarayıcı</h3>', unsafe_allow_html=True)
    st.caption("Tüm BIST 100 hisseleri taranarak değerleme ve momentum bazlı puanlama yapılır.")
    
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        if st.button("🚀 Tüm BIST 100'ü Tara (Algoritmayı Başlat)", use_container_width=True, type="primary"):
            st.session_state.pop("screener_df", None)
            st.cache_data.clear()

    if "screener_df" not in st.session_state:
        results     = []
        prog        = st.progress(0, text="Algoritma BIST 100 hisselerini tarıyor...")
        status_slot = st.empty()

        for i, sym in enumerate(BIST_100):
            status_slot.caption(f"Veri işleniyor: {sym}  ({i+1}/{len(BIST_100)})")
            d = fetch(sym)
            if "error" not in d:
                score = 0
                pv = d["pddd"]
                rv = d["rsi"]
                
                # Değerleme ve Momentum Algoritması
                if isinstance(pv, (int, float)):
                    score += 3 if pv < 1.5 else (1 if pv < 3 else 0)
                if isinstance(rv, (int, float)):
                    score += 3 if rv < 35  else (1 if rv < 45 else 0)
                
                # Trend ve Hacim Bonusu
                if d["trend"] == "Yukselis": score += 1
                if d["vol_surge"] == "Yuksek": score += 1

                results.append({
                    "Hisse":       sym,
                    "Fiyat":       d["price"],
                    "Degisim %":   d["chg"],
                    "RSI":         rv,
                    "PD/DD":       pv if pv else None,
                    "F/K":         d["fk"] if d["fk"] else None,
                    "Sistem Puanı":score,
                })
            prog.progress((i + 1) / len(BIST_100))

        prog.empty()
        status_slot.empty()

        df = (
            pd.DataFrame(results)
            .sort_values("Sistem Puanı", ascending=False)
            .reset_index(drop=True)
        )
        st.session_state.screener_df = df

    df = st.session_state.get("screener_df", pd.DataFrame())

    if not df.empty:
        top = df[df["Sistem Puanı"] >= 4]
        if not top.empty:
            st.success(f"🔥 {len(top)} adet yüksek potansiyelli (Puan ≥ 4) hisse tespit edildi.")
            st.dataframe(top, use_container_width=True, hide_index=True)
            st.divider()

        st.caption("Tüm BIST 100 Sıralaması (Sistem Puanına Göre Azalan)")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("BIST 100 taramasını başlatmak için butona tıklayın.")
