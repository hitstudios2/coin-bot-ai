import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn import hmm
import requests
import json
import os
import warnings
warnings.filterwarnings('ignore')

def get_top_binance_coins(limit=50):
    url = "https://scanner.tradingview.com/crypto/scan"
    post_data = {
        "filter": [
            {"left": "exchange", "operation": "equal", "right": "BINANCE"},
            {"left": "name", "operation": "match", "right": "USDT$"}
        ],
        "sort": {"sortBy": "volume", "sortOrder": "desc"},
        "columns": ["name", "volume"],
        "range": [0, 100]
    }
    headers = {'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'}
    
    try:
        resp = requests.post(url, json=post_data, headers=headers).json()
        data_list = resp.get('data', [])
    except Exception as e:
        print("TradingView API error:", e)
        return []
        
    stable = ['USDCUSDT', 'BUSDUSDT', 'TUSDUSDT', 'FDUSDUSDT', 'EURUSDT', 'TRYUSDT']
    coins = []
    
    for item in data_list:
        sym = item['d'][0]
        if sym not in stable and 'UPUSDT' not in sym and 'DOWNUSDT' not in sym:
            coins.append(sym)
            if len(coins) >= limit:
                break
                
    # yfinance uses -USD
    yf_symbols = [c.replace('USDT', '-USD') for c in coins]
    return yf_symbols

def run_quant_engine():
    print("Fetching top Binance coins...")
    yf_symbols = get_top_binance_coins(50)
    if 'BTC-USD' not in yf_symbols:
        yf_symbols.insert(0, 'BTC-USD')
        
    print(f"Downloading data for {len(yf_symbols)} coins...")
    data = yf.download(yf_symbols, period="180d", interval="1d", progress=False)['Close']
    data.dropna(axis=1, how='all', inplace=True)
    data = data.loc[:, data.count() >= 30]
    data = data.ffill().bfill()
    
    returns = np.log(data / data.shift(1)).dropna()
    
    if 'BTC-USD' not in returns.columns:
        print("Error: BTC-USD not found.")
        return
        
    # HMM Regime Detection on BTC
    btc_rets = returns['BTC-USD']
    btc_vol = btc_rets.rolling(window=20).std().dropna()
    common_idx = btc_rets.index.intersection(btc_vol.index)
    
    features = pd.DataFrame({'returns': btc_rets.loc[common_idx], 'volatility': btc_vol.loc[common_idx]})
    
    hmm_model = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=100, random_state=42)
    hmm_model.fit(features.values)
    
    states = hmm_model.predict(features.values)
    current_state = states[-1]
    
    means = hmm_model.means_
    worst_state = means[:, 0].argmin()
    best_state = means[:, 0].argmax()
    
    if current_state == worst_state:
        regime_label = "GÜÇLÜ DÜŞÜŞ (KRİZ)"
    elif current_state == best_state:
        regime_label = "GÜÇLÜ YÜKSELİŞ"
    elif means[current_state, 0] > 0:
        regime_label = "YÜKSELİŞ (YATAY)"
    else:
        regime_label = "DÜŞÜŞ (YATAY)"
        
    print(f"Current Regime: {regime_label}")
    
    # Calculate Correlation & Score
    recent_returns = returns.tail(30)
    corr_matrix = recent_returns.corr()
    btc_corr = corr_matrix['BTC-USD']
    
    vols = recent_returns.std() * np.sqrt(365)
    min_v, max_v = vols.min(), vols.max()
    if max_v > min_v:
        norm_vols = (vols - min_v) / (max_v - min_v)
    else:
        norm_vols = vols * 0
        
    scores = []
    target_corr = 0.8 if "YÜKSELİŞ" in regime_label else 0.0
    if "KRİZ" in regime_label:
        # Cash defense triggers in Node.js when it sees KRİZ, so scores don't matter as much, but we calculate anyway.
        target_corr = -1.0
        
    for sym in yf_symbols:
        if sym == 'BTC-USD' or sym not in btc_corr:
            continue
            
        c = btc_corr[sym]
        v = norm_vols[sym]
        corr_diff = abs(target_corr - c)
        
        # W1=0.2 (Vol), W2=0.6 (Constant), W3=0.2 (Corr diff) - Best params from WFA
        multi_score = (0.2 * v) + (0.6 * 0.5) - (0.2 * corr_diff)
        
        binance_sym = sym.replace('-USD', 'USDT')
        scores.append({
            "symbol": binance_sym,
            "correlation": float(c),
            "score": float(multi_score)
        })
        
    # Sort by score descending
    scores.sort(key=lambda x: x['score'], reverse=True)
    
    result = {
        "btc_regime": regime_label,
        "scores": scores,
        "updated_at": pd.Timestamp.utcnow().isoformat()
    }
    
    with open("pool_data.json", "w") as f:
        json.dump(result, f, indent=4)
        
    print("Saved pool_data.json successfully.")

if __name__ == "__main__":
    run_quant_engine()
