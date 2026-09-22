# -*- coding: utf-8 -*-
"""
Crypto Quant Master V40 Pro (Dynamic Asset Allocation Engine)
- Dynamic Portfolio Allocation (%) for BTC, Majors, Alts, Cash
- Intuitive Macro State & Tactical Recommendation
- 1-Month (720-Candle) Deep WFO Search
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
# 0. STREAMLIT CONFIG & STYLING
# ============================================================
st.set_page_config(
    page_title="🚀 Crypto Quant Master V40 Pro",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 메인 레이아웃 패딩 최소화 및 카드 CSS 설정
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; color: #0f172a; }
    
    /* 화면 여백 최적화 */
    .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.6rem !important; }
    
    /* 메트릭 박스 여백 압축 */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 8px 12px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    div[data-testid="stMetricLabel"] { font-size: 12px !important; color: #64748b !important; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 20px !important; font-weight: 800; }

    /* 추천 코인 카드 스타일 */
    .card-agg-long {
        background-color: #ffffff; border: 1.5px solid #34d399; border-left: 6px solid #059669;
        padding: 14px; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.03); margin-bottom: 8px;
    }
    .card-agg-short {
        background-color: #ffffff; border: 1.5px solid #f87171; border-left: 6px solid #dc2626;
        padding: 14px; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.03); margin-bottom: 8px;
    }
    .badge-long { background-color: #10b981; color: white; padding: 2px 8px; border-radius: 12px; font-weight: 700; font-size: 11px; }
    .badge-short { background-color: #ef4444; color: white; padding: 2px 8px; border-radius: 12px; font-weight: 700; font-size: 11px; }
    .badge-score { background-color: #6366f1; color: white; padding: 2px 8px; border-radius: 12px; font-weight: 700; font-size: 11px; }
    .tpsl-box { margin-top: 8px; font-size: 12px; color: #334155; background: #f1f5f9; padding: 8px; border-radius: 6px; border: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

DEFAULT_EXCHANGES = ["bitget", "binance", "bybit"]
MAJOR_COINS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "SUI/USDT"]


# ============================================================
# 1. DYNAMIC ASSET ALLOCATION & MACRO ENGINE
# ============================================================
@st.cache_resource(show_spinner=False)
def make_exchange(exchange_id: str):
    cls = getattr(ccxt, exchange_id)
    return cls({"enableRateLimit": True, "timeout": 15000, "options": {"defaultType": "swap"}})


@st.cache_data(ttl=60, show_spinner=False)
def fetch_dynamic_macro_regime() -> tuple[dict, pd.DataFrame, str]:
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

            macro_score = 0
            btc_change = 0.0

            btc_row = df_market[df_market['symbol'] == "BTC/USDT"]
            if not btc_row.empty:
                btc_change = float(btc_row['change_pct'].values[0])

            try:
                btc_symbol = "BTC/USDT" if "BTC/USDT" in ex.markets else "BTC/USDT:USDT"
                btc_ohlcv = ex.fetch_ohlcv(btc_symbol, timeframe="1d", limit=100)
                if len(btc_ohlcv) >= 50:
                    btc_df = pd.DataFrame(btc_ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                    btc_df['ema20'] = ta.trend.EMAIndicator(btc_df['close'], window=20).ema_indicator()
                    btc_df['ema50'] = ta.trend.EMAIndicator(btc_df['close'], window=50).ema_indicator()
                    btc_df['rsi'] = ta.momentum.RSIIndicator(btc_df['close'], window=14).rsi()

                    curr_price = btc_df['close'].iloc[-1]
                    ema20, ema50 = btc_df['ema20'].iloc[-1], btc_df['ema50'].iloc[-1]
                    btc_rsi = btc_df['rsi'].iloc[-1]

                    if curr_price > ema20: macro_score += 35
                    if ema20 > ema50: macro_score += 35
                    if advancing_ratio > 50: macro_score += 20
                    if 45 <= btc_rsi <= 65: macro_score += 10
                else:
                    macro_score = int(advancing_ratio)
            except Exception:
                macro_score = int(np.clip(advancing_ratio + (btc_change * 3), 10, 95))

            # 거시 점수 및 시장 데이터 기반 동적 포트폴리오 비중 연산
            if macro_score >= 70:
                trend_state = "강한 상승장"
                state_badge = "🟢 강한 상승 (BULL)"
                top_target = "💎 메이저 & 🚀 일반 알트"
                btc_pct, major_pct, alt_pct, cash_pct = 20, 45, 35, 0
                action_guide = "🚀 **수익 극대화:** 비트코인 상승 후 순환매가 활발합니다. 알트코인 롱(LONG) 포지션 비중을 늘리세요."
            elif macro_score >= 50:
                trend_state = "완만한 상승장"
                state_badge = "🟢 완만 상승 (BULL)"
                top_target = "₿ 비트코인 & 💎 메이저"
                btc_pct, major_pct, alt_pct, cash_pct = 35, 40, 15, 10
                action_guide = "💡 **주력 자산 집중:** 상승 중기 단계입니다. 변동성이 적은 비트코인 및 메이저 코인 중심으로 안정 운용하세요."
            elif macro_score >= 35:
                trend_state = "박스권 횡보장"
                state_badge = "🟡 박스권/횡보 (NEUTRAL)"
                top_target = "💵 현금 & ₿ BTC 눌림목"
                btc_pct, major_pct, alt_pct, cash_pct = 30, 20, 10, 40
                action_guide = "⚖️ **방어 및 눌림목 관망:** 방향성이 불명확하므로 현금 비중 40% 확보 후 승률 높은 눌림목 타점만 대응하세요."
            else:
                trend_state = "약세/하락장"
                state_badge = "🔴 약세/하락 (BEAR)"
                top_target = "💵 현금 보관 (USDT)"
                btc_pct, major_pct, alt_pct, cash_pct = 10, 10, 0, 80
                action_guide = "🛡️ **자산 방어 최우선:** 매수를 자제하고 현금을 80% 이상 확보하여 위험에 대비하세요."

            return {
                "trend_state": trend_state, "state_badge": state_badge,
                "macro_score": macro_score, "top_target": top_target,
                "btc_pct": btc_pct, "major_pct": major_pct, "alt_pct": alt_pct, "cash_pct": cash_pct,
                "action_guide": action_guide, "advancing_ratio": advancing_ratio,
                "btc_change": btc_change, "market_avg_change": float(df_market['change_pct'].mean())
            }, df_market, ex_id
        except Exception:
            continue

    return {
        "trend_state": "분석 중", "state_badge": "🟡 데이터 연결 중",
        "macro_score": 50, "top_target": "₿ 비트코인 & 💵 현금",
        "btc_pct": 30, "major_pct": 20, "alt_pct": 10, "cash_pct": 40,
        "action_guide": "💡 시황 데이터를 불러오는 중입니다...",
        "advancing_ratio": 50.0, "btc_change": 0.0, "market_avg_change": 0.0
    }, pd.DataFrame(), ""


# ============================================================
# 2. 1-MONTH WFO (720 CANDLES) DEEP ANALYSIS ENGINE
# ============================================================
def analyze_symbol_1month_wfo(symbol: str, exchange_id: str) -> Optional[Dict[str, Any]]:
    try:
        ex = make_exchange(exchange_id)
        raw_symbol = symbol if exchange_id != "binance" else (f"{symbol.replace('/','')}:USDT" if ":" not in symbol else symbol)

        ohlcv = ex.fetch_ohlcv(raw_symbol, timeframe="1h", limit=720)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        if len(df) < 500: return None

        symbol_change_24h = float((df["Close"].iloc[-1] - df["Close"].iloc[-24]) / df["Close"].iloc[-24] * 100)
        if abs(symbol_change_24h) > 20.0: return None

        split_idx = int(len(df) * 0.7)
        df_is = df.iloc[:split_idx].copy()

        ema_windows = [15, 20, 30, 50]
        rsi_bounds = [(40, 65), (45, 70), (35, 60)]

        best_param = None
        best_is_score = -999.0

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

                if trades >= 8:
                    score = pnl / trades
                    if score > best_is_score:
                        best_is_score = score
                        best_param = {"ema": ema_w, "rsi_low": rsi_low, "rsi_high": rsi_high}

        if not best_param or best_is_score <= 0: return None

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

        if oos_win_rate < 50.0 or profit_factor < 1.1:
            return None

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
    st.title("💎 Crypto Quant Master V40 Pro")

    st.sidebar.header("💰 자산 리스크 계산기")
    total_balance = st.sidebar.number_input("내 총 자산 ($)", value=1000.0, step=100.0)
    risk_pct = st.sidebar.slider("1회 매매 허용 리스크 (%)", 0.5, 3.0, 1.0)

    macro_data, market_df, active_ex = fetch_dynamic_macro_regime()
    if market_df.empty:
        st.error("거래소 데이터 로드 실패. 네트워크 상태를 확인해 주세요.")
        return

    # --------------------------------------------------------
    # 🌐 한눈에 보는 컴팩트 거시 분석 대시보드
    # --------------------------------------------------------
    with st.container(border=True):
        # Header Row: 장세 상태 & 가이드
        c_head1, c_head2 = st.columns([1.5, 3])
        with c_head1:
            st.markdown(f"### {macro_data.get('state_badge')}")
            st.caption(f"🎯 집중 타겟: **{macro_data.get('top_target')}**")
        with c_head2:
            st.info(macro_data.get('action_guide'), icon="💡")

        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

        # Main Metrics Row: 자산 배분율(4개) + 핵심 시황 지표(3개)
        m1, m2, m3, m4, m5, m6, m7 = st.columns([1, 1, 1, 1, 1.1, 1.1, 1.1])
        
        m1.metric("₿ 비트코인", f"{macro_data.get('btc_pct', 0)}%")
        m2.metric("💎 메이저", f"{macro_data.get('major_pct', 0)}%")
        m3.metric("🚀 일반알트", f"{macro_data.get('alt_pct', 0)}%")
        m4.metric("💵 현금(USDT)", f"{macro_data.get('cash_pct', 0)}%")
        
        m5.metric("종합 점수", f"{macro_data.get('macro_score', 0)}점")
        m6.metric("BTC 24H", f"{macro_data.get('btc_change', 0.0):+.2f}%")
        m7.metric("상승 종목 비율", f"{macro_data.get('advancing_ratio', 0.0):.1f}%")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 4. 정밀 WFO 스캔 및 타겟 코인 노출
    # --------------------------------------------------------
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

        st.success(f"🎉 스캔 완료! WFO 파라미터 검증을 통과한 정예 롱({len(df_long)}개) / 숏({len(df_short)}개) 코인입니다.")

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
                                <div><span class="badge-long">🟢 LONG</span> &nbsp; <b style="font-size: 16px; color: #0f172a;">{row['symbol']}</b></div>
                                <span class="badge-score">최근 OOS 승률: {row['oos_win_rate']:.1f}%</span>
                            </div>
                            <div class="tpsl-box">
                                ⚙️ <b>1달 최적화:</b> <span style="color:#2563eb; font-weight:700;">{row['opt_param']}</span> | 📊 PF: <b>{row['profit_factor']:.2f}</b><br>
                                💵 현재가: <b>{fmt_price(row['price'])}</b> | 🎯 TP: <span style="color:#059669; font-weight:700;">{fmt_price(row['tp'])}</span> | 🛑 SL: <span style="color:#dc2626; font-weight:700;">{fmt_price(row['sl'])}</span><br>
                                ⚖️ 손익비: 1 : {row['rr_ratio']:.2f} | 💰 권장 진입: <span style="color:#2563eb; font-weight:700;">${pos_usdt:,.1f} USDT</span>
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
                                <div><span class="badge-short">🔴 SHORT</span> &nbsp; <b style="font-size: 16px; color: #0f172a;">{row['symbol']}</b></div>
                                <span class="badge-score">최근 OOS 승률: {row['oos_win_rate']:.1f}%</span>
                            </div>
                            <div class="tpsl-box">
                                ⚙️ <b>1달 최적화:</b> <span style="color:#dc2626; font-weight:700;">{row['opt_param']}</span> | 📊 PF: <b>{row['profit_factor']:.2f}</b><br>
                                💵 현재가: <b>{fmt_price(row['price'])}</b> | 🎯 TP: <span style="color:#059669; font-weight:700;">{fmt_price(row['tp'])}</span> | 🛑 SL: <span style="color:#dc2626; font-weight:700;">{fmt_price(row['sl'])}</span><br>
                                ⚖️ 손익비: 1 : {row['rr_ratio']:.2f} | 💰 권장 진입: <span style="color:#2563eb; font-weight:700;">${pos_usdt:,.1f} USDT</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()