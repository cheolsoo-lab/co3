# -*- coding: utf-8 -*-
"""
Crypto Quant Master Dashboard V40.0 (Visual Analysis Only)
- Clean Light Theme & Intuitive UI
- Automatic Market Regime & Dynamic ATR Scaling
- Quarter-Kelly Risk Sizing Calculator (No Auto-Trading)
"""

from __future__ import annotations

import concurrent.futures
import time
from typing import Optional

import ccxt
import numpy as np
import pandas as pd
import streamlit as st
import ta

# ============================================================
# 0. APP CONFIG & MODERN LIGHT THEME STYLING
# ============================================================
st.set_page_config(
    page_title="🚀 Crypto Quant Master Dashboard V40 Pro",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* 메인 배경: 깨끗하고 부드러운 스노우 화이트 패널 */
    .stApp { background-color: #f8fafc; color: #0f172a; }
    
    /* 거시 마켓 상단 카드: 스카이 블루 & 퍼플 그라데이션 */
    .macro-card {
        background: linear-gradient(135deg, #e0f2fe 0%, #f3e8ff 100%);
        border: 1fr solid #bae6fd;
        padding: 24px; border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05); margin-bottom: 24px;
    }
    
    /* 롱(상승) 시그널 카드: 민트 에메랄드 테마 */
    .card-agg-long {
        background-color: #ffffff;
        border: 2px solid #34d399; border-left: 8px solid #059669;
        padding: 20px; border-radius: 14px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); margin-bottom: 16px;
    }
    
    /* 숏(하락) 시그널 카드: 소프트 로즈 핑크 테마 */
    .card-agg-short {
        background-color: #ffffff;
        border: 2px solid #f87171; border-left: 8px solid #dc2626;
        padding: 20px; border-radius: 14px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); margin-bottom: 16px;
    }
    
    .badge-long { background-color: #10b981; color: white; padding: 6px 12px; border-radius: 20px; font-weight: 800; font-size: 13px; }
    .badge-short { background-color: #ef4444; color: white; padding: 6px 12px; border-radius: 20px; font-weight: 800; font-size: 13px; }
    
    .stat-pill { 
        background: #ffffff; padding: 10px 14px; border-radius: 10px; 
        font-weight: 700; font-size: 14px; color: #1e293b; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.04); text-align: center; border: 1px solid #e2e8f0;
    }
    
    .tpsl-box {
        margin-top: 12px; font-size: 14px; color: #334155; 
        background: #f1f5f9; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

DEFAULT_EXCHANGES = ["bitget", "binance", "bybit"]
MAJOR_COINS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "SUI/USDT"]

SECTOR_MAP = {
    "BTC/USDT": "Macro / L1", "ETH/USDT": "Layer 1", "SOL/USDT": "Layer 1",
    "XRP/USDT": "Payment", "BNB/USDT": "Exchange", "ADA/USDT": "Layer 1",
    "AVAX/USDT": "Layer 1", "SUI/USDT": "Layer 1", "LINK/USDT": "Oracle / DeFi"
}


# ============================================================
# 1. MARKET DATA ENGINE
# ============================================================
@st.cache_resource(show_spinner=False)
def make_exchange(exchange_id: str):
    cls = getattr(ccxt, exchange_id)
    return cls({"enableRateLimit": True, "timeout": 15000, "options": {"defaultType": "swap"}})


@st.cache_data(ttl=60, show_spinner=False)
def fetch_macro_and_market_regime() -> tuple[dict, pd.DataFrame, str]:
    for ex_id in DEFAULT_EXCHANGES:
        try:
            ex = make_exchange(ex_id)
            ex.load_markets()
            tickers = ex.fetch_tickers()
            rows = []
            for symbol, t in tickers.items():
                if not symbol.endswith("USDT") and "/USDT" not in symbol: continue
                clean_symbol = symbol.split(":")[0] if ":" in symbol else symbol
                if not clean_symbol.endswith("/USDT"): continue

                last = t.get("last")
                quote_vol = t.get("quoteVolume") or t.get("baseVolume") or 0.0
                pct = t.get("percentage")
                if last is not None:
                    rows.append({
                        "symbol": clean_symbol,
                        "last": float(last),
                        "change_pct": float(pct) if pct is not None else 0.0,
                        "quote_volume": float(quote_vol) if quote_vol is not None else 0.0,
                    })

            df_market = pd.DataFrame(rows).drop_duplicates("symbol")
            if df_market.empty: continue

            # BTC 거시 시황 분석
            btc_ohlcv = ex.fetch_ohlcv("BTC/USDT", timeframe="1d", limit=60)
            btc_df = pd.DataFrame(btc_ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            btc_df['sma20'] = btc_df['close'].rolling(20).mean()
            btc_df['sma50'] = btc_df['close'].rolling(50).mean()

            curr_price = btc_df['close'].iloc[-1]
            sma20, sma50 = btc_df['sma20'].iloc[-1], btc_df['sma50'].iloc[-1]
            recent_vol = btc_df['volume'].iloc[-5:].mean()
            avg_vol = btc_df['volume'].iloc[-30:].mean()

            if btc_df['close'].iloc[-1] > btc_df['close'].iloc[-5] and recent_vol < avg_vol * 0.9:
                btc_phase = "⚠️ 개미 꼬시기 / 설거지 주의 국면"
            elif curr_price >= btc_df['low'].iloc[-5:].min() and recent_vol >= avg_vol * 0.95:
                btc_phase = "🟢 세력 매집 및 저점 방어 국면"
            else:
                btc_phase = "🔄 물량 소화 및 박스권 다지기"

            if curr_price > sma20 and sma20 > sma50:
                btc_status = "🟢 상승장 (롱 관점 우세)"
                max_risk_ratio = 0.02
            elif curr_price < sma50:
                btc_status = "🔴 하락장 (보수적 접근)"
                max_risk_ratio = 0.005
            else:
                btc_status = "🟡 혼조세 (관망 권장)"
                max_risk_ratio = 0.01

            btc_row = df_market[df_market['symbol'] == "BTC/USDT"]
            btc_change = float(btc_row['change_pct'].values[0]) if not btc_row.empty else 0.0

            return {
                "btc_status": btc_status, "btc_phase": btc_phase,
                "max_risk_ratio": max_risk_ratio, "btc_change": btc_change,
                "market_avg_change": float(df_market['change_pct'].mean())
            }, df_market, ex_id
        except Exception:
            continue
    return {}, pd.DataFrame(), ""


def calculate_volume_profile_poc(df_ohlcv: pd.DataFrame, bins: int = 25) -> float:
    try:
        low_min, high_max = df_ohlcv["Low"].min(), df_ohlcv["High"].max()
        if low_min >= high_max: return float(df_ohlcv["Close"].iloc[-1])
        price_bins = np.linspace(low_min, high_max, bins + 1)
        bin_volumes = np.zeros(bins)
        for _, row in df_ohlcv.iterrows():
            c_low, c_high, c_vol = row["Low"], row["High"], row["Volume"]
            for i in range(bins):
                b_start, b_end = price_bins[i], price_bins[i+1]
                overlap_low, overlap_high = max(c_low, b_start), min(c_high, b_end)
                if overlap_low < overlap_high:
                    fraction = (overlap_high - overlap_low) / (c_high - c_low) if (c_high - c_low) > 0 else 1.0
                    bin_volumes[i] += c_vol * fraction
        return float((price_bins[np.argmax(bin_volumes)] + price_bins[np.argmax(bin_volumes)+1]) / 2.0)
    except Exception:
        return float(df_ohlcv["Close"].iloc[-1])


@st.cache_data(ttl=120, show_spinner=False)
def fetch_dynamic_symbol_data(exchange_id: str, symbol: str) -> Optional[dict]:
    try:
        ex = make_exchange(exchange_id)
        raw_symbol = symbol if exchange_id != "binance" else (f"{symbol.replace('/','')}:USDT" if ":" not in symbol else symbol)

        df_4h = pd.DataFrame(ex.fetch_ohlcv(raw_symbol, timeframe="4h", limit=50), columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df_4h["EMA20"] = ta.trend.EMAIndicator(df_4h["Close"], window=20).ema_indicator()

        df_1h = pd.DataFrame(ex.fetch_ohlcv(raw_symbol, timeframe="1h", limit=60), columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df_1h["EMA20"] = ta.trend.EMAIndicator(df_1h["Close"], window=20).ema_indicator()
        df_1h["RSI14"] = ta.momentum.RSIIndicator(df_1h["Close"], window=14).rsi()
        df_1h["ATR14"] = ta.volatility.AverageTrueRange(df_1h["High"], df_1h["Low"], df_1h["Close"], window=14).average_true_range()
        df_1h["VOL_MA20"] = df_1h["Volume"].rolling(20).mean()
        df_1h["REL_VOLUME"] = df_1h["Volume"] / df_1h["VOL_MA20"]

        if len(df_4h) < 20 or len(df_1h) < 20: return None
        return {"df_4h": df_4h, "df_1h": df_1h, "poc": calculate_volume_profile_poc(df_1h)}
    except Exception:
        return None


# ============================================================
# 2. QUANT ANALYSIS & SIGNAL SCANNER
# ============================================================
def analyze_symbol(symbol: str, exchange_id: str, market_avg_change: float) -> Optional[dict]:
    data = fetch_dynamic_symbol_data(exchange_id.lower(), symbol)
    if not data: return None

    df_4h, df_1h, poc = data["df_4h"], data["df_1h"], data["poc"]
    r_4h, r_1h = df_4h.iloc[-1], df_1h.iloc[-1]
    close = float(r_1h["Close"])
    atr = float(r_1h["ATR14"]) if pd.notna(r_1h["ATR14"]) and r_1h["ATR14"] > 0 else close * 0.02
    rsi_1h = float(r_1h["RSI14"]) if pd.notna(r_1h["RSI14"]) else 50.0
    rel_vol = float(r_1h["REL_VOLUME"]) if pd.notna(r_1h["REL_VOLUME"]) else 1.0

    symbol_change_24h = float((close - df_1h.iloc[-24]["Close"]) / df_1h.iloc[-24]["Close"] * 100) if len(df_1h) >= 24 else 0.0
    if abs(symbol_change_24h) > 12.0: return None  # 뇌동매매 방지

    is_4h_long = float(r_4h["Close"]) >= float(r_4h["EMA20"]) * 0.995
    is_4h_short = float(r_4h["Close"]) <= float(r_4h["EMA20"]) * 1.005

    group, pos_type = None, None
    if is_4h_long and float(r_1h["Close"]) >= float(r_1h["EMA20"]) * 0.995 and 42 <= rsi_1h <= 65:
        group, pos_type = "AGGRESSIVE", "LONG"
    elif is_4h_short and float(r_1h["Close"]) <= float(r_1h["EMA20"]) * 1.005 and 35 <= rsi_1h <= 58:
        group, pos_type = "AGGRESSIVE", "SHORT"
    elif is_4h_long and 45 <= rsi_1h <= 60:
        group, pos_type = "STABLE", "LONG"
    elif is_4h_short and 40 <= rsi_1h <= 55:
        group, pos_type = "STABLE", "SHORT"
    else:
        return None

    if pos_type == "LONG":
        sl = max(close - (2.2 * atr), poc * 0.985)
        risk = close - sl
        tp = max(close + (2.5 * atr), close + (risk * 2.8))
    else:
        sl = min(close + (2.2 * atr), poc * 1.015)
        risk = sl - close
        tp = min(close - (2.5 * atr), close - (risk * 2.8))

    risk_val, reward_val = abs(close - sl), abs(tp - close)
    if risk_val <= 0 or (reward_val / risk_val) < 1.5: return None

    return {
        "symbol": symbol, "price": close, "group": group, "pos_type": pos_type,
        "tp": float(tp), "sl": float(sl), "rr_ratio": reward_val / risk_val,
        "score": float(np.clip(rel_vol * 20 + abs(symbol_change_24h - market_avg_change) * 10, 50, 100))
    }


def fmt_price(x):
    if x is None or x <= 0: return "$0.00"
    if x >= 1000: return f"${x:,.2f}"
    if x >= 1: return f"${x:,.4f}"
    return f"${x:,.8f}"


# ============================================================
# 3. STREAMLIT UI MAIN
# ============================================================
def main():
    st.title("💎 Crypto Quant Master Dashboard V40 Pro")
    st.caption("차트 분석 & 리스크 계산기 전용 버전에 오신 것을 환영합니다.")

    # 사이드바: 자산 입력 기반 진입 추천 비중 계산기
    st.sidebar.header("💰 자산 리스크 계산기")
    total_balance = st.sidebar.number_input("내 총 자산 ($)", value=1000.0, step=100.0)
    risk_pct = st.sidebar.slider("1회 매매 허용 리스크 (%)", 0.5, 3.0, 1.0)
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Quarter-Kelly 리스크 제어:**\n내 자산의 1~2% 이상의 손실 위험을 감수하지 않도록 최적 진입 금액을 계산합니다.")

    macro_data, market_df, active_ex = fetch_macro_and_market_regime()
    if market_df.empty:
        st.error("거래소 연결 실패. 네트워크 상태를 확인해 주세요.")
        return

    # 거시 시황 비주얼 판넬
    st.markdown(f"""
    <div class="macro-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <h2 style="margin: 0; color: #0f172a; font-size: 20px;">🌐 마켓 상태 및 세력 수급 보고서</h2>
            <span style="background: #e0e7ff; color: #3730a3; padding: 8px 16px; border-radius: 20px; font-weight: 800; font-size: 15px;">
                {macro_data.get('btc_status', '분석 중')}
            </span>
        </div>
        <p style="font-size: 15px; color: #475569; margin-bottom: 16px;">
            • 비트코인 세력 수급: <b>{macro_data.get('btc_phase', '-')}</b><br>
            • 추천 최대 위험 노출도: <b>자산의 {macro_data.get('max_risk_ratio', 0.01)*100:.1f}% 이내 권장</b>
        </p>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;">
            <div class="stat-pill">₿ BTC 24H: <span style="color:#0284c7;">{macro_data.get('btc_change', 0.0):+.2f}%</span></div>
            <div class="stat-pill">🎯 검색 대상: <b>Top 100 우량주</b></div>
            <div class="stat-pill">🏦 연동 데이터: <b>{active_ex.upper()} 선물</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    top_volume_market = market_df.sort_values(by="quote_volume", ascending=False).head(100)
    available_majors = [s for s in MAJOR_COINS if s in market_df["symbol"].values]
    symbols = list(set(available_majors + top_volume_market["symbol"].tolist()))

    if st.button("🔍 우량 알트코인 매매 타점 스캔 시작", use_container_width=True, type="primary"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_symbols = len(symbols)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(analyze_symbol, s, active_ex, macro_data["market_avg_change"]): s for s in symbols}
            completed = 0
            for f in concurrent.futures.as_completed(futures):
                completed += 1
                progress_bar.progress(completed / total_symbols)
                status_text.text(f"⚡ 차트 분석 및 매물대 스캔 중... ({completed}/{total_symbols})")
                r = f.result()
                if r: results.append(r)

        progress_bar.empty()
        status_text.empty()
        st.session_state["v40_view_results"] = results

    results = st.session_state.get("v40_view_results", [])
    if results:
        df_res = pd.DataFrame(results)
        agg_df = df_res[df_res["group"] == "AGGRESSIVE"].sort_values("score", ascending=False)
        stable_df = df_res[df_res["group"] == "STABLE"].sort_values("score", ascending=False)

        st.success(f"🎉 스캔 완료! 총 {len(df_res)}개의 진입 추천 포지션이 발굴되었습니다.")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### 🔥 공격형 모멘텀 타점 ({len(agg_df)}개)")
            for _, row in agg_df.iterrows():
                is_long = row["pos_type"] == "LONG"
                card_cls = "card-agg-long" if is_long else "card-agg-short"
                badge = '<span class="badge-long">🟢 LONG (매수)</span>' if is_long else '<span class="badge-short">🔴 SHORT (매도)</span>'
                
                # 리스크 금액 기반 권장 진입 금액 계산
                sl_pct = abs(row['price'] - row['sl']) / row['price']
                risk_usdt = total_balance * (risk_pct / 100.0)
                pos_usdt = risk_usdt / sl_pct if sl_pct > 0 else 0

                st.markdown(f"""
                <div class="{card_cls}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>{badge} &nbsp; <b style="font-size: 18px; color: #0f172a;">{row['symbol']}</b></div>
                        <div style="font-size: 16px;">현재가: <b>{fmt_price(row['price'])}</b></div>
                    </div>
                    <div class="tpsl-box">
                        🎯 **목표 익절가(TP):** <span style="color:#059669; font-weight:700;">{fmt_price(row['tp'])}</span><br>
                        🛑 **손절 기준가(SL):** <span style="color:#dc2626; font-weight:700;">{fmt_price(row['sl'])}</span><br>
                        ⚖️ **손익비:** 1 : {row['rr_ratio']:.2f}<br>
                        💵 **권장 진입 규모:** <span style="color:#2563eb; font-weight:700;">${pos_usdt:,.1f} USDT</span> (리스크: ${risk_usdt:,.1f})
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"### 🛡️ 안정형 스윙 타점 ({len(stable_df)}개)")
            for _, row in stable_df.iterrows():
                is_long = row["pos_type"] == "LONG"
                card_cls = "card-agg-long" if is_long else "card-agg-short"
                badge = '<span class="badge-long">🟢 LONG (매수)</span>' if is_long else '<span class="badge-short">🔴 SHORT (매도)</span>'

                sl_pct = abs(row['price'] - row['sl']) / row['price']
                risk_usdt = total_balance * (risk_pct / 100.0)
                pos_usdt = risk_usdt / sl_pct if sl_pct > 0 else 0

                st.markdown(f"""
                <div class="{card_cls}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>{badge} &nbsp; <b style="font-size: 18px; color: #0f172a;">{row['symbol']}</b></div>
                        <div style="font-size: 16px;">현재가: <b>{fmt_price(row['price'])}</b></div>
                    </div>
                    <div class="tpsl-box">
                        🎯 **목표 익절가(TP):** <span style="color:#059669; font-weight:700;">{fmt_price(row['tp'])}</span><br>
                        🛑 **손절 기준가(SL):** <span style="color:#dc2626; font-weight:700;">{fmt_price(row['sl'])}</span><br>
                        ⚖️ **손익비:** 1 : {row['rr_ratio']:.2f}<br>
                        💵 **권장 진입 규모:** <span style="color:#2563eb; font-weight:700;">${pos_usdt:,.1f} USDT</span> (리스크: ${risk_usdt:,.1f})
                    </div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()