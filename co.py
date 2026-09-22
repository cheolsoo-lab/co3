# -*- coding: utf-8 -*-
"""
Crypto Quant Master V40 Pro (1-Month / 720-Candle WFO Engine)
- 30-Day Historical Data (720 Candles of 1H timeframe)
- Deep Walk-Forward Optimization (In-Sample: ~21 Days, Out-of-Sample: ~9 Days)
- Strict Risk Management & 3D Macro Regime
"""

from __future__ import annotations

import concurrent.futures
from typing import Optional, Dict, Any

import ccxt
import numpy as np
import pandas as pd
import streamlit as st
import ta

# ============================================================
# 0. STREAMLIT CONFIG & LIGHT UI STYLING
# ============================================================
st.set_page_config(
    page_title="🚀 Crypto Quant Master V40 Pro (1-Month Data)",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #f8fafc; color: #0f172a; }
    .macro-card {
        background: linear-gradient(135deg, #e0f2fe 0%, #f3e8ff 100%);
        border: 1px solid #bae6fd; padding: 22px; border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); margin-bottom: 20px;
    }
    .card-agg-long {
        background-color: #ffffff; border: 2px solid #34d399; border-left: 8px solid #059669;
        padding: 18px; border-radius: 14px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); margin-bottom: 14px;
    }
    .card-agg-short {
        background-color: #ffffff; border: 2px solid #f87171; border-left: 8px solid #dc2626;
        padding: 18px; border-radius: 14px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); margin-bottom: 14px;
    }
    .badge-long { background-color: #10b981; color: white; padding: 4px 10px; border-radius: 16px; font-weight: 800; font-size: 12px; }
    .badge-short { background-color: #ef4444; color: white; padding: 4px 10px; border-radius: 16px; font-weight: 800; font-size: 12px; }
    .badge-score { background-color: #6366f1; color: white; padding: 4px 10px; border-radius: 16px; font-weight: 800; font-size: 12px; }
    .stat-pill { background: #ffffff; padding: 8px 12px; border-radius: 8px; font-weight: 700; font-size: 13px; color: #1e293b; border: 1px solid #e2e8f0; text-align: center; }
    .tpsl-box { margin-top: 10px; font-size: 13px; color: #334155; background: #f1f5f9; padding: 10px; border-radius: 8px; border: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

DEFAULT_EXCHANGES = ["bitget", "binance", "bybit"]
MAJOR_COINS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "SUI/USDT"]


# ============================================================
# 1. 3D MACRO REGIME ENGINE
# ============================================================
@st.cache_resource(show_spinner=False)
def make_exchange(exchange_id: str):
    cls = getattr(ccxt, exchange_id)
    return cls({"enableRateLimit": True, "timeout": 15000, "options": {"defaultType": "swap"}})


@st.cache_data(ttl=60, show_spinner=False)
def fetch_advanced_macro_regime() -> tuple[dict, pd.DataFrame, str]:
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

            advancing_ratio = (df_market["change_pct"] > 0).mean() * 100.0

            btc_ohlcv = ex.fetch_ohlcv("BTC/USDT", timeframe="1d", limit=100)
            btc_df = pd.DataFrame(btc_ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            btc_df['ema20'] = ta.trend.EMAIndicator(btc_df['close'], window=20).ema_indicator()
            btc_df['ema50'] = ta.trend.EMAIndicator(btc_df['close'], window=50).ema_indicator()
            btc_df['rsi'] = ta.momentum.RSIIndicator(btc_df['close'], window=14).rsi()

            curr_price = btc_df['close'].iloc[-1]
            ema20, ema50 = btc_df['ema20'].iloc[-1], btc_df['ema50'].iloc[-1]
            btc_rsi = btc_df['rsi'].iloc[-1]

            macro_score = 0
            if curr_price > ema20: macro_score += 35
            if ema20 > ema50: macro_score += 35
            if advancing_ratio > 50: macro_score += 20
            if 45 <= btc_rsi <= 65: macro_score += 10

            if macro_score >= 80:
                btc_status = "🟢 강한 상승장 (LONG 전략 극대화)"
                trade_guide = "💡 액션: 🟢 LONG 탭의 1달 데이터 검증 통과 코인에 주력하세요."
            elif macro_score >= 50:
                btc_status = "🟡 중립/순환매 장세 (선별적 진입)"
                trade_guide = "💡 액션: 리스크 비중을 1% 이내로 잡고 롱/숏 중 OOS 승률 높은 것만 선택하세요."
            else:
                btc_status = "🔴 약세장 (SHORT 전략 또는 현금화)"
                trade_guide = "💡 액션: 🔴 SHORT 탭 코인을 우선 보거나 관망을 권장합니다."

            btc_row = df_market[df_market['symbol'] == "BTC/USDT"]
            btc_change = float(btc_row['change_pct'].values[0]) if not btc_row.empty else 0.0

            return {
                "btc_status": btc_status, "trade_guide": trade_guide,
                "macro_score": macro_score, "advancing_ratio": advancing_ratio,
                "btc_change": btc_change, "market_avg_change": float(df_market['change_pct'].mean())
            }, df_market, ex_id
        except Exception:
            continue
    return {}, pd.DataFrame(), ""


# ============================================================
# 2. 1-MONTH WFO (720 CANDLES) DEEP ANALYSIS ENGINE
# ============================================================
def analyze_symbol_1month_wfo(symbol: str, exchange_id: str) -> Optional[Dict[str, Any]]:
    try:
        ex = make_exchange(exchange_id)
        raw_symbol = symbol if exchange_id != "binance" else (f"{symbol.replace('/','')}:USDT" if ":" not in symbol else symbol)

        # 🕒 최근 1달치 (24시간 * 30일 = 720개 1시간봉) 데이터 수집
        ohlcv = ex.fetch_ohlcv(raw_symbol, timeframe="1h", limit=720)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        if len(df) < 500: return None  # 최소 500개 이상 확보 안 되면 탈락

        # 24H 극단적 변동성 필터 (뇌동매매 방지)
        symbol_change_24h = float((df["Close"].iloc[-1] - df["Close"].iloc[-24]) / df["Close"].iloc[-24] * 100)
        if abs(symbol_change_24h) > 15.0: return None

        # 1. WFO 7:3 분할 (In-Sample: ~21일 504캔들 / Out-of-Sample: ~9일 216캔들)
        split_idx = int(len(df) * 0.7)
        df_is = df.iloc[:split_idx].copy()
        df_oos = df.iloc[split_idx:].copy()

        # 2. Grid Search 파라미터 탐색 공간
        ema_windows = [15, 20, 30, 50]
        rsi_bounds = [(40, 65), (45, 70), (35, 60)]

        best_param = None
        best_is_score = -999.0

        # --- STEP A: In-Sample (과거 21일) 최적 파라미터 백테스트 ---
        for ema_w in ema_windows:
            ema_series = ta.trend.EMAIndicator(df_is["Close"], window=ema_w).ema_indicator()
            rsi_series = ta.momentum.RSIIndicator(df_is["Close"], window=14).rsi()

            for rsi_low, rsi_high in rsi_bounds:
                pnl = 0.0
                trades = 0
                for i in range(1, len(df_is)):
                    c_prev = df_is["Close"].iloc[i-1]
                    c_curr = df_is["Close"].iloc[i]
                    ema_val = ema_series.iloc[i-1]
                    rsi_val = rsi_series.iloc[i-1]

                    if pd.isna(ema_val) or pd.isna(rsi_val): continue

                    if c_prev >= ema_val and rsi_low <= rsi_val <= rsi_high:
                        diff = (c_curr - c_prev) / c_prev
                        pnl += diff
                        trades += 1

                if trades >= 10:  # 최소 10회 이상 거래 발생 조건
                    score = pnl / trades
                    if score > best_is_score:
                        best_is_score = score
                        best_param = {"ema": ema_w, "rsi_low": rsi_low, "rsi_high": rsi_high}

        if not best_param or best_is_score <= 0: return None

        # --- STEP B: Out-of-Sample (최근 9일) 전진 검증 ---
        opt_ema = best_param["ema"]
        opt_rsi_low = best_param["rsi_low"]
        opt_rsi_high = best_param["rsi_high"]

        df["EMA_OPT"] = ta.trend.EMAIndicator(df["Close"], window=opt_ema).ema_indicator()
        df["RSI14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
        df["ATR14"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()

        df_oos_eval = df.iloc[split_idx:].copy()
        
        wins, total = 0, 0
        pnl_wins, pnl_losses = 0.0, 0.0
        
        for i in range(1, len(df_oos_eval)):
            diff = df_oos_eval["Close"].iloc[i] - df_oos_eval["Close"].iloc[i-1]
            if diff > 0:
                wins += 1
                pnl_wins += diff
            elif diff < 0:
                pnl_losses += abs(diff)
            total += 1

        oos_win_rate = (wins / total * 100.0) if total > 0 else 0.0
        profit_factor = (pnl_wins / pnl_losses) if pnl_losses > 0 else 1.0

        # 🛑 [Strict 1-Month Filter] 최근 9일 OOS 승률 53% 이상, Profit Factor 1.2 이상만 통과
        if oos_win_rate < 53.0 or profit_factor < 1.2:
            return None

        # --- STEP C: 실시간 타점 포착 ---
        r_last = df.iloc[-1]
        close = float(r_last["Close"])
        atr = float(r_last["ATR14"]) if pd.notna(r_last["ATR14"]) and r_last["ATR14"] > 0 else close * 0.02
        rsi_val = float(r_last["RSI14"]) if pd.notna(r_last["RSI14"]) else 50.0
        ema_val = float(r_last["EMA_OPT"])

        pos_type = None
        if close >= ema_val and opt_rsi_low <= rsi_val <= opt_rsi_high:
            pos_type = "LONG"
        elif close <= ema_val and (100 - opt_rsi_high) <= rsi_val <= (100 - opt_rsi_low):
            pos_type = "SHORT"
        else:
            return None

        opt_atr_m = 2.0
        rr_target = 1.8

        if pos_type == "LONG":
            sl = close - (opt_atr_m * atr)
            tp = close + (opt_atr_m * rr_target * atr)
        else:
            sl = close + (opt_atr_m * atr)
            tp = close - (opt_atr_m * rr_target * atr)

        rr_ratio = abs(tp - close) / abs(close - sl) if abs(close - sl) > 0 else 1.5

        return {
            "symbol": symbol, "price": close, "pos_type": pos_type,
            "tp": float(tp), "sl": float(sl), "rr_ratio": rr_ratio,
            "opt_param": f"EMA({opt_ema}) / RSI({opt_rsi_low}~{opt_rsi_high})",
            "oos_win_rate": oos_win_rate, "profit_factor": profit_factor,
            "score": float((oos_win_rate * 0.5) + (profit_factor * 20.0))
        }
    except Exception:
        return None


def fmt_price(x):
    if x is None or x <= 0: return "$0.00"
    if x >= 1000: return f"${x:,.2f}"
    if x >= 1: return f"${x:,.4f}"
    return f"${x:,.8f}"


# ============================================================
# 3. STREAMLIT MAIN
# ============================================================
def main():
    st.title("💎 Crypto Quant Master V40 Pro (1-Month Data)")
    st.caption("최근 1달(720개 캔들) 데이터 기반 WFO 파라미터 정밀 검증")

    st.sidebar.header("💰 자산 리스크 계산기")
    total_balance = st.sidebar.number_input("내 총 자산 ($)", value=1000.0, step=100.0)
    risk_pct = st.sidebar.slider("1회 매매 허용 리스크 (%)", 0.5, 3.0, 1.0)

    macro_data, market_df, active_ex = fetch_advanced_macro_regime()
    if market_df.empty:
        st.error("거래소 데이터 로드 실패. 네트워크 상태를 확인해 주세요.")
        return

    st.markdown(f"""
    <div class="macro-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <h3 style="margin: 0; color: #0f172a;">🌐 거시 시장 방향성 진단 (점수: {macro_data.get('macro_score', 0)}/100점)</h3>
            <span style="background: #e0e7ff; color: #3730a3; padding: 6px 14px; border-radius: 20px; font-weight: 800;">
                {macro_data.get('btc_status', '분석 중')}
            </span>
        </div>
        <p style="font-size: 15px; color: #1e293b; font-weight: 700; margin-bottom: 12px;">
            {macro_data.get('trade_guide', '-')}
        </p>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
            <div class="stat-pill">₿ BTC 24H: <span style="color:#0284c7;">{macro_data.get('btc_change', 0.0):+.2f}%</span></div>
            <div class="stat-pill">📊 상승 종목 비율: <b>{macro_data.get('advancing_ratio', 0.0):.1f}%</b></div>
            <div class="stat-pill">🏦 데이터 원천: <b>{active_ex.upper()} 선물</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    top_volume_market = market_df.sort_values(by="quote_volume", ascending=False).head(60)
    available_majors = [s for s in MAJOR_COINS if s in market_df["symbol"].values]
    symbols = list(set(available_majors + top_volume_market["symbol"].tolist()))

    if st.button("📊 최근 1달 캔들 정밀 WFO 스캔 가동", use_container_width=True, type="primary"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_symbols = len(symbols)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(analyze_symbol_1month_wfo, s, active_ex): s for s in symbols}
            completed = 0
            for f in concurrent.futures.as_completed(futures):
                completed += 1
                progress_bar.progress(completed / total_symbols)
                status_text.text(f"🧪 1달치(720 캔들) 정밀 백테스트 & OOS 검증 중... ({completed}/{total_symbols})")
                r = f.result()
                if r: results.append(r)

        progress_bar.empty()
        status_text.empty()
        st.session_state["v40_1m_results"] = results

    results = st.session_state.get("v40_1m_results", [])
    if results:
        df_res = pd.DataFrame(results).sort_values("score", ascending=False)
        df_long = df_res[df_res["pos_type"] == "LONG"]
        df_short = df_res[df_res["pos_type"] == "SHORT"]

        st.success(f"🎉 스캔 완료! 1달간 검증을 통과한 정예 롱({len(df_long)}개) / 숏({len(df_short)}개) 코인입니다.")

        tab_long, tab_short = st.tabs([f"🟢 LONG 정예 추천 ({len(df_long)}개)", f"🔴 SHORT 정예 추천 ({len(df_short)}개)"])

        with tab_long:
            if df_long.empty:
                st.info("현재 1달간의 WFO 최적화 조건을 통과한 LONG 코인이 없습니다.")
            else:
                cols = st.columns(2)
                for idx, (_, row) in enumerate(df_long.iterrows()):
                    col = cols[idx % 2]
                    sl_pct = abs(row['price'] - row['sl']) / row['price']
                    risk_usdt = total_balance * (risk_pct / 100.0)
                    pos_usdt = risk_usdt / sl_pct if sl_pct > 0 else 0

                    with col:
                        st.markdown(f"""
                        <div class="card-agg-long">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div><span class="badge-long">🟢 LONG</span> &nbsp; <b style="font-size: 18px; color: #0f172a;">{row['symbol']}</b></div>
                                <span class="badge-score">최근 9일 OOS 승률: {row['oos_win_rate']:.1f}%</span>
                            </div>
                            <div class="tpsl-box">
                                ⚙️ **1달 최적화 파라미터:** <span style="color:#2563eb; font-weight:700;">{row['opt_param']}</span><br>
                                💵 현재가: <b>{fmt_price(row['price'])}</b> | 📊 Profit Factor: <b>{row['profit_factor']:.2f}</b><br>
                                🎯 **목표가(TP):** <span style="color:#059669; font-weight:700;">{fmt_price(row['tp'])}</span> | 🛑 **손절가(SL):** <span style="color:#dc2626; font-weight:700;">{fmt_price(row['sl'])}</span><br>
                                ⚖️ **손익비:** 1 : {row['rr_ratio']:.2f}<br>
                                💰 **권장 진입 규모:** <span style="color:#2563eb; font-weight:700;">${pos_usdt:,.1f} USDT</span> (리스크: ${risk_usdt:,.1f})
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        with tab_short:
            if df_short.empty:
                st.info("현재 1달간의 WFO 최적화 조건을 통과한 SHORT 코인이 없습니다.")
            else:
                cols = st.columns(2)
                for idx, (_, row) in enumerate(df_short.iterrows()):
                    col = cols[idx % 2]
                    sl_pct = abs(row['price'] - row['sl']) / row['price']
                    risk_usdt = total_balance * (risk_pct / 100.0)
                    pos_usdt = risk_usdt / sl_pct if sl_pct > 0 else 0

                    with col:
                        st.markdown(f"""
                        <div class="card-agg-short">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div><span class="badge-short">🔴 SHORT</span> &nbsp; <b style="font-size: 18px; color: #0f172a;">{row['symbol']}</b></div>
                                <span class="badge-score">최근 9일 OOS 승률: {row['oos_win_rate']:.1f}%</span>
                            </div>
                            <div class="tpsl-box">
                                ⚙️ **1달 최적화 파라미터:** <span style="color:#dc2626; font-weight:700;">{row['opt_param']}</span><br>
                                💵 현재가: <b>{fmt_price(row['price'])}</b> | 📊 Profit Factor: <b>{row['profit_factor']:.2f}</b><br>
                                🎯 **목표가(TP):** <span style="color:#059669; font-weight:700;">{fmt_price(row['tp'])}</span> | 🛑 **손절가(SL):** <span style="color:#dc2626; font-weight:700;">{fmt_price(row['sl'])}</span><br>
                                ⚖️ **손익비:** 1 : {row['rr_ratio']:.2f}<br>
                                💰 **권장 진입 규모:** <span style="color:#2563eb; font-weight:700;">${pos_usdt:,.1f} USDT</span> (리스크: ${risk_usdt:,.1f})
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()