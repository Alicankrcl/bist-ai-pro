"""
ml_engine.py — BIST Pro Terminal Makine Öğrenmesi Modülü
Gerçek geçmiş veri üzerinde anlık eğitim yapan, hafif sıklet RandomForest Classifier.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ham OHLCV DataFrame'inden teknik özellikler (feature) üretir."""
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    feat = pd.DataFrame(index=df.index)

    # ── YARDIMCI: Wilder's Smoothing (RMA) ───────────────────────────────────
    def rma(series, length):
        return series.ewm(alpha=1/length, adjust=False).mean()

    # ── RSI(14) ──────────────────────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = -delta.where(delta < 0, 0.0)
    avg_gain = rma(gain, 14)
    avg_loss = rma(loss, 14)
    feat["rsi"] = 100 - (100 / (1 + (avg_gain / avg_loss)))

    # ── MACD(12,26,9) ────────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    feat["macd"]      = macd
    feat["macd_hist"] = macd - sig   # Histogram: negatiften pozitife geçiş = alım sinyali

    # ── Bollinger Band Konumu ─────────────────────────────────────────────────
    sma20  = close.rolling(20).mean()
    std20  = close.rolling(20).std()
    upper  = sma20 + 2 * std20
    lower  = sma20 - 2 * std20
    feat["bb_pos"] = (close - lower) / (upper - lower + 1e-9)  # 0=alt bant, 1=üst bant

    # ── ATR(14) — Volatilite ──────────────────────────────────────────────────
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = rma(tr, 14)
    feat["atr_pct"] = atr / close    # Fiyata göre normalize ATR (%)

    # ── EMA Eğimi & Cross ────────────────────────────────────────────────────
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    feat["ema_cross"] = (ema20 - ema50) / close   # Pozitif = bullish cross

    # ── Momentum (son 5 ve 10 gün) ────────────────────────────────────────────
    feat["mom5"]  = close.pct_change(5)
    feat["mom10"] = close.pct_change(10)

    # ── Hacim Anomalisi ──────────────────────────────────────────────────────
    vol_ma20 = vol.rolling(20).mean()
    feat["vol_ratio"] = vol / (vol_ma20 + 1e-9)

    # ── Fiyat / SMA20 Oranı ───────────────────────────────────────────────────
    feat["price_sma_ratio"] = close / (sma20 + 1e-9)

    return feat


def run_ml_prediction(df: pd.DataFrame) -> dict:
    """
    Geçmiş veriyle RandomForestClassifier eğitir ve son satır için
    yükseliş olasılığı + güven endeksi döner.

    Döndürülen dict:
        prob_up   : 0-100 arası yükseliş olasılığı (%)
        prob_down : 0-100 arası düşüş olasılığı (%)
        signal    : "GÜÇLÜ ALIŞ" / "ALIŞ" / "NÖTR" / "SATIŞ" / "GÜÇLÜ SATIŞ"
        confidence: 0-100 arası model güven skoru (en yüksek sınıf olasılığı)
        n_trees   : Kullanılan ağaç sayısı
        accuracy  : CV doğruluğu (yaklaşık, son fold)
        error     : Hata mesajı (başarılıysa None)
    """
    try:
        if len(df) < 80:
            return {"error": "Yeterli geçmiş veri yok (min 80 gün)"}

        # Etiket: ertesi gün kapanış > bugünkü kapanış ise 1 (Yükseliş), değilse 0
        df = df.copy()
        df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

        feat = _compute_features(df)
        feat["target"] = df["target"]
        feat = feat.dropna()

        if len(feat) < 60:
            return {"error": "Temizleme sonrası veri yetersiz"}

        feature_cols = [c for c in feat.columns if c != "target"]
        X = feat[feature_cols].values
        y = feat["target"].values

        # Son satır tahmin için ayrılır, geri kalanı eğitim
        X_train, X_last = X[:-1], X[[-1]]
        y_train         = y[:-1]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_last  = scaler.transform(X_last)

        # Model: 200 ağaç, hafif ama sağlam
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )

        # Zaman serisi çapraz doğrulama (son fold doğruluğu)
        tscv = TimeSeriesSplit(n_splits=4)
        fold_acc = []
        for tr_idx, val_idx in tscv.split(X_train):
            model.fit(X_train[tr_idx], y_train[tr_idx])
            preds = model.predict(X_train[val_idx])
            fold_acc.append((preds == y_train[val_idx]).mean())

        cv_accuracy = float(np.mean(fold_acc))

        # Son eğitim (tüm eğitim seti) ve tahmin
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_last)[0]   # [P(düşüş), P(yükseliş)]
        prob_up   = round(proba[1] * 100, 1)
        prob_down = round(proba[0] * 100, 1)
        confidence = round(max(proba) * 100, 1)

        # Sinyal etiketi
        if prob_up >= 70:
            signal = "GÜÇLÜ ALIŞ"
        elif prob_up >= 55:
            signal = "ALIŞ"
        elif prob_up <= 30:
            signal = "GÜÇLÜ SATIŞ"
        elif prob_up <= 45:
            signal = "SATIŞ"
        else:
            signal = "NÖTR"

        return {
            "prob_up":    prob_up,
            "prob_down":  prob_down,
            "signal":     signal,
            "confidence": confidence,
            "n_trees":    200,
            "accuracy":   round(cv_accuracy * 100, 1),
            "error":      None,
        }

    except Exception as e:
        return {"error": str(e)}
