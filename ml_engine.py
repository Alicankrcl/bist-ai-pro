"""
ml_engine.py — BIST Pro Terminal Makine Öğrenmesi Modülü
Hedge-fon standartlarında zaman serisi modeli (Random Forest) ve özellik mühendisliği.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame'inden endüstri standardı teknik özellikleri hesaplar."""
    close = df['Close']
    high  = df['High']
    low   = df['Low']
    vol   = df['Volume']

    feat = pd.DataFrame(index=df.index)

    # ── Wilder's Smoothing (RMA) YARDIMCISI ──
    def rma(series, length):
        return series.ewm(alpha=1/length, adjust=False).mean()

    # ── 1. RSI(14) ──
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = -delta.where(delta < 0, 0.0)
    avg_gain = rma(gain, 14)
    avg_loss = rma(loss, 14)
    feat['rsi'] = 100 - (100 / (1 + (avg_gain / avg_loss)))

    # ── 2. MACD(12,26,9) ──
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    feat['macd'] = ema12 - ema26
    feat['macd_sig'] = feat['macd'].ewm(span=9, adjust=False).mean()

    # ── 3. EMA(20) & EMA(50) ──
    feat['ema20'] = close.ewm(span=20, adjust=False).mean()
    feat['ema50'] = close.ewm(span=50, adjust=False).mean()

    # ── 4. Bollinger Bantları (20, 2) ──
    sma20  = close.rolling(20).mean()
    std20  = close.rolling(20).std()
    feat['bb_upper'] = sma20 + 2 * std20
    feat['bb_lower'] = sma20 - 2 * std20

    # ── 5. ATR(14) ──
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    feat['atr'] = rma(tr, 14)

    # ── 6. Hacim Faktörü (vol_factor) ──
    vol_ma20 = vol.rolling(20).mean()
    feat['vol_factor'] = vol / (vol_ma20 + 1e-9)
    
    # ── Fiyat Sütunu (Skorlama İçin İhtiyaç Var) ──
    feat['price'] = close

    return feat


def run_ml_prediction(df: pd.DataFrame) -> dict:
    """
    RandomForestClassifier eğitir ve tahmin döndürür.
    Zaman serisi sızıntısını önleyen katı test/eğitim ayrımına sahiptir.
    """
    neutral_dict = {
        'prob_up': 50.0,
        'prob_down': 50.0,
        'signal': 'NEUTRAL',
        'confidence': 50.0,
        'n_trees': 300,
        'accuracy': 50.0,
        'error': 'Yetersiz veri veya model hatası'
    }
    
    try:
        if df is None or len(df) < 50:
            return neutral_dict

        # Etiket: ertesi gün kapanış > bugünkü kapanış ise 1 (Yükseliş), değilse 0
        df = df.copy()
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(float)

        feat = _compute_features(df)
        feat['target'] = df['target']
        
        # Son satır bizim tahmin edeceğimiz (bugünkü) satırdır
        last_row = feat.iloc[[-1]].copy()
        
        # Eğitim seti için son satırı çıkarıp NaN'ları düşüyoruz
        train_feat = feat.iloc[:-1].dropna()

        if len(train_feat) < 50:
            neutral_dict['error'] = 'Temizleme sonrası yeterli eğitim verisi kalmadı (min 50)'
            return neutral_dict

        feature_cols = [c for c in feat.columns if c not in ['target', 'price']]
        
        X_train = train_feat[feature_cols].values
        y_train = train_feat['target'].values
        X_last  = last_row[feature_cols].values

        # Ölçeklendirme
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        # X_last içinde nan varsa median/0 ile dolduralım çökmeyi önlemek için
        X_last = np.nan_to_num(X_last, nan=0.0)
        X_last  = scaler.transform(X_last)

        # Model: 300 ağaç, dengeli sınıf ağırlıkları, max_depth=8
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight='balanced',
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

        cv_accuracy = float(np.mean(fold_acc)) if fold_acc else 50.0

        # Son eğitim (tüm eğitim seti) ve tahmin
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_last)[0]   # [P(düşüş), P(yükseliş)]
        
        # Sınıf sıralamasını garantiye alalım (0: düşüş, 1: yükseliş)
        classes = list(model.classes_)
        if 1.0 in classes:
            idx_up = classes.index(1.0)
            idx_down = classes.index(0.0)
            prob_up = round(proba[idx_up] * 100, 1)
            prob_down = round(proba[idx_down] * 100, 1)
        else:
            prob_up, prob_down = 50.0, 50.0
            
        confidence = round(max(prob_up, prob_down), 1)

        # Seçici Sinyal etiketi (Eşik >= %60)
        if prob_up >= 60.0:
            signal = 'BUY'
        elif prob_down >= 60.0:
            signal = 'SELL'
        else:
            signal = 'NEUTRAL'

        return {
            'prob_up':    prob_up,
            'prob_down':  prob_down,
            'signal':     signal,
            'confidence': confidence,
            'n_trees':    300,
            'accuracy':   round(cv_accuracy * 100, 1),
            'error':      None,
        }

    except Exception as e:
        neutral_dict['error'] = str(e)
        return neutral_dict
