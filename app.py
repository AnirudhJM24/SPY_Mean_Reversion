"""
Volatility Mean-Reversion Options Strategy
==========================================
Streamlit dashboard that:
1. Pulls real SPY options data + VIX/SPY history via yfinance
2. Builds the implied vol smile/surface
3. Identifies rich/cheap vol regions → constructs a mean-reversion strategy
4. Calibrates the Heston model to the live smile
5. Monte Carlo simulates strategy P&L under Heston dynamics
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings("ignore")

from heston import (
    bs_price, implied_vol_from_price, calibrate_heston,
    heston_implied_vol, simulate_heston_paths, price_strategy_mc
)

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="Vol Mean-Reversion Strategy",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16192b 100%);
        border: 1px solid #2d3348;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }
    .metric-label {
        color: #8b92a8;
        font-size: 13px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        color: #e2e8f0;
        font-size: 28px;
        font-weight: 700;
        margin-top: 4px;
    }
    .section-header {
        color: #c9d1d9;
        font-size: 22px;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 2px solid #f0883e;
    }
    .strategy-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        font-family: 'Courier New', monospace;
        color: #c9d1d9;
    }
</style>
""", unsafe_allow_html=True)

st.title("📉 Volatility Mean-Reversion Strategy Engine")
st.markdown("*Exploiting implied vol mean-reversion on SPY via Heston simulation*")

# =============================================================================
# Sidebar controls
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    ticker_symbol = st.text_input("Ticker", value="SPY")
    lookback_days = st.slider("Historical Lookback (days)", 60, 504, 252)
    rv_window = st.slider("Realized Vol Window (days)", 10, 60, 30)
    n_mc_paths = st.select_slider("Monte Carlo Paths",
                                   options=[1000, 5000, 10000, 25000, 50000],
                                   value=10000)
    vol_threshold = st.slider("Vol Richness Threshold (σ)", 0.5, 3.0, 1.5, 0.1,
                               help="How many std devs above/below mean to flag as rich/cheap")

    st.markdown("---")
    st.markdown("**Strategy Type**")
    strategy_type = st.radio("", ["Vega-Neutral Spread", "Short Straddle + Wing Hedge",
                                   "Put Ratio Spread"], index=0)

    run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)


# =============================================================================
# Data loading
# =============================================================================
@st.cache_data(ttl=300)
def load_market_data(ticker, lookback):
    """Pull historical prices, VIX, and options chain."""
    stock = yf.Ticker(ticker)

    # Historical prices
    end = datetime.now()
    start = end - timedelta(days=int(lookback * 1.5))
    hist = stock.history(start=start, end=end)

    # VIX history
    vix = yf.Ticker("^VIX")
    vix_hist = vix.history(start=start, end=end)

    # Options chain — grab multiple expiries
    try:
        expiry_dates = stock.options[:6]  # up to 6 nearest expiries
    except:
        expiry_dates = []

    chains = {}
    for exp in expiry_dates:
        try:
            chain = stock.option_chain(exp)
            chains[exp] = {
                'calls': chain.calls,
                'puts': chain.puts
            }
        except:
            continue

    # Current price
    current_price = hist['Close'].iloc[-1]

    # Risk-free rate approximation (13-week T-bill via ^IRX)
    try:
        irx = yf.Ticker("^IRX")
        rf_hist = irx.history(period="5d")
        risk_free = rf_hist['Close'].iloc[-1] / 100.0
    except:
        risk_free = 0.045  # fallback

    return hist, vix_hist, chains, expiry_dates, current_price, risk_free


@st.cache_data(ttl=300)
def compute_realized_vol(prices, window):
    """Annualized realized volatility from log returns."""
    log_ret = np.log(prices / prices.shift(1)).dropna()
    rv = log_ret.rolling(window=window).std() * np.sqrt(252)
    return rv


def build_vol_smile(chain_calls, chain_puts, S, T, r):
    """Extract implied vols from an options chain for a single expiry."""
    smile_data = []

    for _, row in chain_calls.iterrows():
        K = row['strike']
        mid = (row['bid'] + row['ask']) / 2 if row['bid'] > 0 and row['ask'] > 0 else row['lastPrice']
        if mid <= 0 or T <= 0:
            continue
        # Use impliedVolatility from yfinance if available, else compute
        iv = row.get('impliedVolatility', None)
        if iv is None or iv <= 0 or iv > 3.0:
            iv = implied_vol_from_price(mid, S, K, T, r, 'call')
        if iv and not np.isnan(iv) and 0.01 < iv < 3.0:
            smile_data.append({
                'K': K, 'T': T, 'iv': iv, 'type': 'call',
                'moneyness': K / S, 'mid_price': mid
            })

    for _, row in chain_puts.iterrows():
        K = row['strike']
        mid = (row['bid'] + row['ask']) / 2 if row['bid'] > 0 and row['ask'] > 0 else row['lastPrice']
        if mid <= 0 or T <= 0:
            continue
        iv = row.get('impliedVolatility', None)
        if iv is None or iv <= 0 or iv > 3.0:
            iv = implied_vol_from_price(mid, S, K, T, r, 'put')
        if iv and not np.isnan(iv) and 0.01 < iv < 3.0:
            smile_data.append({
                'K': K, 'T': T, 'iv': iv, 'type': 'put',
                'moneyness': K / S, 'mid_price': mid
            })

    return smile_data


def construct_strategy(smile_df, S, T, r, strategy_type, vol_threshold, rv_mean):
    """
    Identify rich/cheap vol regions and construct the strategy.
    Returns list of strategy legs.
    """
    if smile_df.empty:
        return [], {}

    # Find ATM IV
    smile_df['abs_moneyness'] = abs(smile_df['moneyness'] - 1.0)
    atm_row = smile_df.loc[smile_df['abs_moneyness'].idxmin()]
    atm_iv = atm_row['iv']

    # Compute vol richness: how far each strike's IV is from realized vol mean
    smile_df['vol_richness'] = (smile_df['iv'] - rv_mean) / smile_df['iv'].std()

    # Rich options: IV significantly above RV mean (sell these)
    rich = smile_df[smile_df['vol_richness'] > vol_threshold].copy()
    # Cheap options: IV below or near RV mean (buy these)
    cheap = smile_df[smile_df['vol_richness'] < -vol_threshold * 0.5].copy()

    legs = []
    info = {
        'atm_iv': atm_iv,
        'rv_mean': rv_mean,
        'rich_strikes': rich['K'].tolist() if not rich.empty else [],
        'cheap_strikes': cheap['K'].tolist() if not cheap.empty else [],
    }

    if strategy_type == "Vega-Neutral Spread":
        # Short ATM put (richest vol region) + long OTM put (wing hedge)
        # + Short ATM call + long OTM call
        # Creates an iron condor / iron butterfly shaped payoff

        # Short the richest put
        puts = smile_df[smile_df['type'] == 'put'].sort_values('vol_richness', ascending=False)
        calls = smile_df[smile_df['type'] == 'call'].sort_values('vol_richness', ascending=False)

        if len(puts) >= 2 and len(calls) >= 2:
            # Short the richest options
            short_put = puts.iloc[0]
            short_call = calls.iloc[0]

            # Long cheap wing options for protection
            long_put = puts.iloc[-1]  # cheapest vol put
            long_call = calls.iloc[-1]  # cheapest vol call

            # Ensure proper ordering
            if long_put['K'] < short_put['K'] and long_call['K'] > short_call['K']:
                legs = [
                    {'type': 'put', 'K': long_put['K'], 'position': 1,
                     'premium': long_put['mid_price'], 'iv': long_put['iv']},
                    {'type': 'put', 'K': short_put['K'], 'position': -1,
                     'premium': short_put['mid_price'], 'iv': short_put['iv']},
                    {'type': 'call', 'K': short_call['K'], 'position': -1,
                     'premium': short_call['mid_price'], 'iv': short_call['iv']},
                    {'type': 'call', 'K': long_call['K'], 'position': 1,
                     'premium': long_call['mid_price'], 'iv': long_call['iv']},
                ]

    elif strategy_type == "Short Straddle + Wing Hedge":
        # Short ATM straddle (sell rich vol) + buy OTM strangle (hedge tails)
        calls = smile_df[smile_df['type'] == 'call'].copy()
        puts = smile_df[smile_df['type'] == 'put'].copy()

        atm_call = calls.loc[calls['abs_moneyness'].idxmin()]
        atm_put = puts.loc[puts['abs_moneyness'].idxmin()]

        # OTM wings at ~5% out
        otm_calls = calls[calls['moneyness'] > 1.04].sort_values('moneyness')
        otm_puts = puts[puts['moneyness'] < 0.96].sort_values('moneyness', ascending=False)

        if not otm_calls.empty and not otm_puts.empty:
            wing_call = otm_calls.iloc[0]
            wing_put = otm_puts.iloc[0]
            legs = [
                {'type': 'put', 'K': atm_put['K'], 'position': -1,
                 'premium': atm_put['mid_price'], 'iv': atm_put['iv']},
                {'type': 'call', 'K': atm_call['K'], 'position': -1,
                 'premium': atm_call['mid_price'], 'iv': atm_call['iv']},
                {'type': 'put', 'K': wing_put['K'], 'position': 1,
                 'premium': wing_put['mid_price'], 'iv': wing_put['iv']},
                {'type': 'call', 'K': wing_call['K'], 'position': 1,
                 'premium': wing_call['mid_price'], 'iv': wing_call['iv']},
            ]

    elif strategy_type == "Put Ratio Spread":
        # Sell 2 rich OTM puts, buy 1 ATM put — classic skew trade
        puts = smile_df[smile_df['type'] == 'put'].copy()
        atm_put = puts.loc[puts['abs_moneyness'].idxmin()]
        otm_puts = puts[puts['moneyness'] < 0.96].sort_values('vol_richness', ascending=False)

        if not otm_puts.empty:
            rich_otm = otm_puts.iloc[0]
            legs = [
                {'type': 'put', 'K': atm_put['K'], 'position': 1,
                 'premium': atm_put['mid_price'], 'iv': atm_put['iv']},
                {'type': 'put', 'K': rich_otm['K'], 'position': -2,
                 'premium': rich_otm['mid_price'], 'iv': rich_otm['iv']},
            ]

    # Fallback: if no legs found, create a basic iron condor
    if not legs:
        calls = smile_df[smile_df['type'] == 'call'].sort_values('K')
        puts = smile_df[smile_df['type'] == 'put'].sort_values('K')
        if len(calls) >= 2 and len(puts) >= 2:
            atm_idx_c = (calls['moneyness'] - 1.0).abs().idxmin()
            atm_idx_p = (puts['moneyness'] - 1.0).abs().idxmin()
            legs = [
                {'type': 'put', 'K': puts['K'].iloc[0], 'position': 1,
                 'premium': puts['mid_price'].iloc[0], 'iv': puts['iv'].iloc[0]},
                {'type': 'put', 'K': puts.loc[atm_idx_p, 'K'], 'position': -1,
                 'premium': puts.loc[atm_idx_p, 'mid_price'], 'iv': puts.loc[atm_idx_p, 'iv']},
                {'type': 'call', 'K': calls.loc[atm_idx_c, 'K'], 'position': -1,
                 'premium': calls.loc[atm_idx_c, 'mid_price'], 'iv': calls.loc[atm_idx_c, 'iv']},
                {'type': 'call', 'K': calls['K'].iloc[-1], 'position': 1,
                 'premium': calls['mid_price'].iloc[-1], 'iv': calls['iv'].iloc[-1]},
            ]

    return legs, info


# =============================================================================
# Main execution
# =============================================================================

if run_button:
    with st.spinner("Fetching market data..."):
        try:
            hist, vix_hist, chains, expiry_dates, S, r = load_market_data(
                ticker_symbol, lookback_days
            )
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            st.stop()

    if hist.empty:
        st.error("No historical data retrieved. Check ticker symbol.")
        st.stop()

    if not chains:
        st.error("No options data available for this ticker.")
        st.stop()

    # =========================================================================
    # Section 1: Historical Vol Analysis — Mean Reversion Evidence
    # =========================================================================
    st.markdown('<div class="section-header">1 │ Volatility Mean Reversion</div>',
                unsafe_allow_html=True)

    rv = compute_realized_vol(hist['Close'], rv_window)
    rv_mean = rv.dropna().mean()
    rv_std = rv.dropna().std()
    rv_current = rv.dropna().iloc[-1]

    # Align VIX with price data
    vix_series = vix_hist['Close'].reindex(hist.index, method='ffill') / 100.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Current Price</div>
            <div class="metric-value">${S:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">{rv_window}d Realized Vol</div>
            <div class="metric-value">{rv_current:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        vix_current = vix_series.dropna().iloc[-1] if not vix_series.dropna().empty else 0
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">VIX (Implied Vol)</div>
            <div class="metric-value">{vix_current:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        vol_premium = vix_current - rv_current
        color = "#f85149" if vol_premium > 0 else "#3fb950"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Vol Risk Premium</div>
            <div class="metric-value" style="color:{color}">{vol_premium:+.1%}</div>
        </div>""", unsafe_allow_html=True)

    # Plot: IV vs RV with mean reversion bands
    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1],
                                        facecolor='#0e1117')
    for ax in [ax1a, ax1b]:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Top: IV and RV time series
    rv_plot = rv.dropna()
    ax1a.plot(rv_plot.index, rv_plot.values, color='#58a6ff', linewidth=1.5,
              label=f'{rv_window}d Realized Vol', alpha=0.9)

    vix_plot = vix_series.dropna()
    if not vix_plot.empty:
        ax1a.plot(vix_plot.index, vix_plot.values, color='#f0883e', linewidth=1.5,
                  label='VIX (Implied Vol)', alpha=0.9)

    ax1a.axhline(rv_mean, color='#3fb950', linestyle='--', alpha=0.7, label=f'RV Mean ({rv_mean:.1%})')
    ax1a.fill_between(rv_plot.index,
                      rv_mean - rv_std, rv_mean + rv_std,
                      alpha=0.1, color='#3fb950', label='±1σ Band')
    ax1a.fill_between(rv_plot.index,
                      rv_mean - 2 * rv_std, rv_mean + 2 * rv_std,
                      alpha=0.05, color='#3fb950', label='±2σ Band')

    ax1a.set_ylabel('Annualized Volatility', color='#c9d1d9', fontsize=11)
    ax1a.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1a.legend(loc='upper right', fontsize=9, facecolor='#161b22', edgecolor='#30363d',
                labelcolor='#c9d1d9')
    ax1a.set_title('Implied vs Realized Volatility — Mean Reversion', color='#c9d1d9',
                   fontsize=14, fontweight='bold', pad=12)

    # Bottom: Vol spread (IV - RV)
    if not vix_plot.empty:
        common_idx = rv_plot.index.intersection(vix_plot.index)
        if len(common_idx) > 0:
            spread = vix_plot.loc[common_idx] - rv_plot.loc[common_idx]
            colors_spread = ['#f85149' if x > 0 else '#3fb950' for x in spread]
            ax1b.bar(common_idx, spread, color=colors_spread, alpha=0.6, width=1.5)
            ax1b.axhline(0, color='#8b949e', linewidth=0.5)
            ax1b.set_ylabel('IV − RV Spread', color='#c9d1d9', fontsize=11)
            ax1b.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    plt.tight_layout()
    st.pyplot(fig1)

    st.markdown("""
    > **Interpretation:** When IV (orange) is significantly above RV (blue) and the mean (green dashed),
    > implied vol is *rich* — historically it tends to revert down. This is the edge we exploit by selling vol.
    """)

    # =========================================================================
    # Section 2: Vol Smile / Surface
    # =========================================================================
    st.markdown('<div class="section-header">2 │ Implied Volatility Smile & Surface</div>',
                unsafe_allow_html=True)

    all_smile_data = []
    expiry_smiles = {}

    for exp_str in expiry_dates:
        if exp_str not in chains:
            continue
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
        T = max((exp_date - datetime.now()).days / 365.0, 0.001)
        if T < 0.01 or T > 2.0:
            continue

        smile = build_vol_smile(chains[exp_str]['calls'], chains[exp_str]['puts'], S, T, r)
        if smile:
            all_smile_data.extend(smile)
            expiry_smiles[exp_str] = pd.DataFrame(smile)

    if not all_smile_data:
        st.warning("Could not build vol smile from options data.")
        st.stop()

    smile_df_all = pd.DataFrame(all_smile_data)

    # Filter to reasonable moneyness range
    smile_df_all = smile_df_all[
        (smile_df_all['moneyness'] > 0.85) & (smile_df_all['moneyness'] < 1.15)
    ].copy()

    # Plot smile for each expiry
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0e1117')
    for ax in [ax2a, ax2b]:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    colors_exp = ['#58a6ff', '#f0883e', '#d2a8ff', '#3fb950', '#f85149', '#e3b341']
    for i, (exp_str, df_exp) in enumerate(expiry_smiles.items()):
        df_plot = df_exp[(df_exp['moneyness'] > 0.88) & (df_exp['moneyness'] < 1.12)]
        if df_plot.empty:
            continue
        df_plot = df_plot.sort_values('moneyness')
        # Average calls and puts at same strike
        df_avg = df_plot.groupby('moneyness').agg({'iv': 'mean'}).reset_index()
        ax2a.plot(df_avg['moneyness'], df_avg['iv'],
                  color=colors_exp[i % len(colors_exp)],
                  linewidth=2, marker='o', markersize=3, label=exp_str, alpha=0.85)

    ax2a.axvline(1.0, color='#8b949e', linestyle=':', alpha=0.5, label='ATM')
    ax2a.axhline(rv_mean, color='#3fb950', linestyle='--', alpha=0.5, label=f'RV Mean')
    ax2a.set_xlabel('Moneyness (K/S)', color='#c9d1d9')
    ax2a.set_ylabel('Implied Volatility', color='#c9d1d9')
    ax2a.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2a.set_title('Vol Smile by Expiry', color='#c9d1d9', fontweight='bold')
    ax2a.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')

    # Vol surface (3D-ish heatmap)
    if len(smile_df_all) > 20:
        moneyness_grid = np.linspace(0.88, 1.12, 50)
        T_grid = np.linspace(smile_df_all['T'].min(), smile_df_all['T'].max(), 50)
        M_mesh, T_mesh = np.meshgrid(moneyness_grid, T_grid)

        try:
            iv_surface = griddata(
                (smile_df_all['moneyness'].values, smile_df_all['T'].values),
                smile_df_all['iv'].values,
                (M_mesh, T_mesh),
                method='cubic'
            )
            c = ax2b.pcolormesh(M_mesh, T_mesh * 365, iv_surface,
                                cmap='magma', shading='auto')
            plt.colorbar(c, ax=ax2b, label='IV', format=mticker.PercentFormatter(1.0))
            ax2b.set_xlabel('Moneyness (K/S)', color='#c9d1d9')
            ax2b.set_ylabel('Days to Expiry', color='#c9d1d9')
            ax2b.set_title('Vol Surface', color='#c9d1d9', fontweight='bold')
        except:
            ax2b.text(0.5, 0.5, 'Insufficient data\nfor surface interpolation',
                      ha='center', va='center', color='#8b949e', fontsize=12,
                      transform=ax2b.transAxes)

    plt.tight_layout()
    st.pyplot(fig2)

    # =========================================================================
    # Section 3: Strategy Construction
    # =========================================================================
    st.markdown('<div class="section-header">3 │ Mean-Reversion Strategy</div>',
                unsafe_allow_html=True)

    # Use the nearest expiry with enough data for the strategy
    target_exp = None
    target_smile_df = None
    target_T = None
    for exp_str in expiry_dates:
        if exp_str in expiry_smiles and len(expiry_smiles[exp_str]) >= 10:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            T_candidate = (exp_date - datetime.now()).days / 365.0
            if 0.03 < T_candidate < 0.25:  # 1 week to 3 months
                target_exp = exp_str
                target_smile_df = expiry_smiles[exp_str].copy()
                target_T = T_candidate
                break

    if target_smile_df is None:
        # fallback to first available
        for exp_str in expiry_dates:
            if exp_str in expiry_smiles and len(expiry_smiles[exp_str]) >= 6:
                target_exp = exp_str
                target_smile_df = expiry_smiles[exp_str].copy()
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
                target_T = max((exp_date - datetime.now()).days / 365.0, 0.01)
                break

    if target_smile_df is None:
        st.error("Not enough options data to construct strategy.")
        st.stop()

    # Filter to reasonable range
    target_smile_df = target_smile_df[
        (target_smile_df['moneyness'] > 0.90) & (target_smile_df['moneyness'] < 1.10)
    ].copy()

    legs, strategy_info = construct_strategy(
        target_smile_df, S, target_T, r, strategy_type, vol_threshold, rv_mean
    )

    if not legs:
        st.warning("Could not construct strategy with current data. Try adjusting the vol threshold.")
        st.stop()

    st.markdown(f"**Target Expiry:** {target_exp} ({int(target_T*365)} DTE)")
    st.markdown(f"**Strategy:** {strategy_type}")

    # Display legs
    leg_data = []
    net_premium = 0
    for leg in legs:
        pos_str = f"{'Long' if leg['position'] > 0 else 'Short'} {abs(leg['position'])}"
        leg_data.append({
            'Position': pos_str,
            'Type': leg['type'].upper(),
            'Strike': f"${leg['K']:.0f}",
            'Premium': f"${leg['premium']:.2f}",
            'IV': f"{leg['iv']:.1%}",
            'Moneyness': f"{leg['K']/S:.3f}"
        })
        net_premium += leg['position'] * leg['premium']

    st.dataframe(pd.DataFrame(leg_data), use_container_width=True, hide_index=True)

    credit_debit = "CREDIT" if net_premium < 0 else "DEBIT"
    st.markdown(f"**Net Premium:** ${abs(net_premium):.2f} ({credit_debit})")

    # Strategy payoff diagram
    fig3, ax3 = plt.subplots(figsize=(14, 4), facecolor='#0e1117')
    ax3.set_facecolor('#161b22')
    ax3.tick_params(colors='#8b949e')
    ax3.spines['bottom'].set_color('#30363d')
    ax3.spines['left'].set_color('#30363d')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    S_range = np.linspace(S * 0.85, S * 1.15, 500)
    payoff = np.zeros_like(S_range)
    for leg in legs:
        if leg['type'] == 'call':
            payoff += leg['position'] * (np.maximum(S_range - leg['K'], 0) - leg['premium'])
        else:
            payoff += leg['position'] * (np.maximum(leg['K'] - S_range, 0) - leg['premium'])

    ax3.fill_between(S_range, payoff, 0, where=(payoff > 0), alpha=0.3, color='#3fb950')
    ax3.fill_between(S_range, payoff, 0, where=(payoff < 0), alpha=0.3, color='#f85149')
    ax3.plot(S_range, payoff, color='#e2e8f0', linewidth=2)
    ax3.axhline(0, color='#8b949e', linewidth=0.5)
    ax3.axvline(S, color='#f0883e', linestyle=':', alpha=0.7, label=f'Spot (${S:.0f})')

    for leg in legs:
        ax3.axvline(leg['K'], color='#58a6ff', linestyle='--', alpha=0.3)

    ax3.set_xlabel(f'{ticker_symbol} Price at Expiry', color='#c9d1d9')
    ax3.set_ylabel('P&L per Contract ($)', color='#c9d1d9')
    ax3.set_title('Strategy Payoff at Expiry', color='#c9d1d9', fontweight='bold')
    ax3.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    plt.tight_layout()
    st.pyplot(fig3)

    # =========================================================================
    # Section 4: Heston Calibration
    # =========================================================================
    st.markdown('<div class="section-header">4 │ Heston Model Calibration</div>',
                unsafe_allow_html=True)

    with st.spinner("Calibrating Heston model to market smile... (this may take 30-60s)"):
        # Prepare calibration data — use calls only for cleaner calibration
        calib_data = []
        for _, row in smile_df_all.iterrows():
            if row['type'] == 'call' and 0.90 < row['moneyness'] < 1.10:
                calib_data.append({'K': row['K'], 'T': row['T'], 'iv': row['iv']})

        if len(calib_data) < 5:
            # Include puts too
            for _, row in smile_df_all.iterrows():
                if 0.90 < row['moneyness'] < 1.10:
                    calib_data.append({'K': row['K'], 'T': row['T'], 'iv': row['iv']})

        # Deduplicate
        seen = set()
        unique_calib = []
        for d in calib_data:
            key = (round(d['K'], 1), round(d['T'], 4))
            if key not in seen:
                seen.add(key)
                unique_calib.append(d)

        if len(unique_calib) < 5:
            st.error("Not enough data points for Heston calibration.")
            st.stop()

        heston_params = calibrate_heston(unique_calib, S, r)

    # Display parameters
    col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns(5)
    param_cols = [col_h1, col_h2, col_h3, col_h4, col_h5]
    param_names = ['v₀', 'κ (MR Speed)', 'θ (Long-Run Var)', 'σᵥ (Vol of Vol)', 'ρ (Correlation)']
    param_keys = ['v0', 'kappa', 'theta', 'sigma_v', 'rho']
    param_fmts = ['.4f', '.2f', '.4f', '.4f', '.4f']

    for col, name, key, fmt in zip(param_cols, param_names, param_keys, param_fmts):
        val = heston_params[key]
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{name}</div>
                <div class="metric-value" style="font-size:22px">{val:{fmt}}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"**Calibration RMSE:** {heston_params['rmse']:.4f}")

    # Half-life of vol mean reversion
    kappa = heston_params['kappa']
    half_life = np.log(2) / kappa * 252  # in trading days
    st.markdown(f"**Vol Mean-Reversion Half-Life:** {half_life:.0f} trading days ({half_life/21:.1f} months)")

    # Show model vs market fit
    fig4, ax4 = plt.subplots(figsize=(14, 4), facecolor='#0e1117')
    ax4.set_facecolor('#161b22')
    ax4.tick_params(colors='#8b949e')
    ax4.spines['bottom'].set_color('#30363d')
    ax4.spines['left'].set_color('#30363d')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # Get unique expiries for comparison
    unique_T = sorted(set(round(d['T'], 4) for d in unique_calib))
    for i, T_val in enumerate(unique_T[:4]):
        this_exp = [d for d in unique_calib if abs(d['T'] - T_val) < 0.005]
        if len(this_exp) < 3:
            continue
        this_exp.sort(key=lambda x: x['K'])
        market_K = [d['K'] / S for d in this_exp]
        market_iv = [d['iv'] for d in this_exp]
        model_iv = [heston_implied_vol(S, d['K'], d['T'], r,
                                        heston_params['v0'], heston_params['kappa'],
                                        heston_params['theta'], heston_params['sigma_v'],
                                        heston_params['rho'])
                    for d in this_exp]

        c = colors_exp[i % len(colors_exp)]
        dte = int(T_val * 365)
        ax4.scatter(market_K, market_iv, color=c, s=25, alpha=0.7, zorder=5,
                    label=f'Market ({dte}d)')
        valid = [(k, m) for k, m in zip(market_K, model_iv) if m and not np.isnan(m)]
        if valid:
            ax4.plot([v[0] for v in valid], [v[1] for v in valid],
                     color=c, linewidth=2, alpha=0.8, linestyle='--')

    ax4.set_xlabel('Moneyness (K/S)', color='#c9d1d9')
    ax4.set_ylabel('Implied Volatility', color='#c9d1d9')
    ax4.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax4.set_title('Heston Fit vs Market (dots = market, dashes = model)', color='#c9d1d9',
                  fontweight='bold')
    ax4.legend(fontsize=9, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    plt.tight_layout()
    st.pyplot(fig4)

    # =========================================================================
    # Section 5: Monte Carlo Simulation
    # =========================================================================
    st.markdown('<div class="section-header">5 │ Heston Monte Carlo — Strategy P&L</div>',
                unsafe_allow_html=True)

    with st.spinner(f"Simulating {n_mc_paths:,} paths under Heston dynamics..."):
        np.random.seed(42)
        n_steps = max(int(target_T * 252), 20)
        S_paths, v_paths = simulate_heston_paths(
            S, heston_params['v0'], heston_params['kappa'], heston_params['theta'],
            heston_params['sigma_v'], heston_params['rho'], r, target_T,
            n_paths=n_mc_paths, n_steps=n_steps
        )

        pnl = price_strategy_mc(S_paths, v_paths, legs, r, target_T)

    # P&L Statistics
    mean_pnl = np.mean(pnl)
    median_pnl = np.median(pnl)
    std_pnl = np.std(pnl)
    sharpe = mean_pnl / std_pnl * np.sqrt(252 / (target_T * 252)) if std_pnl > 0 else 0
    win_rate = np.mean(pnl > 0)
    max_loss = np.min(pnl)
    max_gain = np.max(pnl)
    var_95 = np.percentile(pnl, 5)
    cvar_95 = np.mean(pnl[pnl <= var_95])

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        color = "#3fb950" if mean_pnl > 0 else "#f85149"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Expected P&L</div>
            <div class="metric-value" style="color:{color}">${mean_pnl:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with col_s2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value">{win_rate:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with col_s3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value">{sharpe:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with col_s4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">95% CVaR</div>
            <div class="metric-value" style="color:#f85149">${cvar_95:.2f}</div>
        </div>""", unsafe_allow_html=True)

    # P&L Distribution
    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0e1117')
    for ax in [ax5a, ax5b]:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Histogram
    n_bins = 80
    counts, bins, patches = ax5a.hist(pnl, bins=n_bins, alpha=0.8, edgecolor='none')
    for patch, b in zip(patches, bins[:-1]):
        patch.set_facecolor('#3fb950' if b >= 0 else '#f85149')

    ax5a.axvline(mean_pnl, color='#f0883e', linewidth=2, linestyle='-', label=f'Mean ${mean_pnl:.2f}')
    ax5a.axvline(var_95, color='#d2a8ff', linewidth=1.5, linestyle='--', label=f'5% VaR ${var_95:.2f}')
    ax5a.axvline(0, color='#e2e8f0', linewidth=0.8, linestyle=':')
    ax5a.set_xlabel('P&L ($)', color='#c9d1d9')
    ax5a.set_ylabel('Frequency', color='#c9d1d9')
    ax5a.set_title('Strategy P&L Distribution', color='#c9d1d9', fontweight='bold')
    ax5a.legend(fontsize=9, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')

    # Sample paths
    n_show = min(50, n_mc_paths)
    t_axis = np.linspace(0, target_T * 365, n_steps + 1)
    for i in range(n_show):
        ax5b.plot(t_axis, S_paths[i, :], color='#58a6ff', alpha=0.08, linewidth=0.5)
    ax5b.plot(t_axis, np.mean(S_paths[:, :], axis=0), color='#f0883e', linewidth=2,
              label='Mean Path')
    ax5b.plot(t_axis, np.percentile(S_paths[:, :], 5, axis=0), color='#f85149',
              linewidth=1, linestyle='--', label='5th Percentile')
    ax5b.plot(t_axis, np.percentile(S_paths[:, :], 95, axis=0), color='#3fb950',
              linewidth=1, linestyle='--', label='95th Percentile')

    for leg in legs:
        ax5b.axhline(leg['K'], color='#d2a8ff', linestyle=':', alpha=0.4)

    ax5b.set_xlabel('Days', color='#c9d1d9')
    ax5b.set_ylabel(f'{ticker_symbol} Price', color='#c9d1d9')
    ax5b.set_title('Simulated Price Paths (Heston)', color='#c9d1d9', fontweight='bold')
    ax5b.legend(fontsize=9, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    plt.tight_layout()
    st.pyplot(fig5)

    # =========================================================================
    # Section 6: Detailed Stats Table
    # =========================================================================
    st.markdown('<div class="section-header">6 │ Full Results Summary</div>',
                unsafe_allow_html=True)

    summary = pd.DataFrame({
        'Metric': [
            'Underlying', 'Spot Price', 'Risk-Free Rate', 'Strategy', 'Expiry', 'DTE',
            '', 'Heston v₀', 'Heston κ', 'Heston θ', 'Heston σᵥ', 'Heston ρ',
            'MR Half-Life (days)', 'Calibration RMSE',
            '', 'ATM IV', f'{rv_window}d Realized Vol', 'Vol Risk Premium',
            '', 'Net Premium', 'E[P&L]', 'Median P&L', 'Std Dev P&L',
            'Win Rate', 'Max Gain', 'Max Loss', '95% VaR', '95% CVaR',
            'Annualized Sharpe'
        ],
        'Value': [
            ticker_symbol, f'${S:.2f}', f'{r:.2%}', strategy_type, target_exp,
            f'{int(target_T*365)}',
            '─── Heston Parameters ───',
            f'{heston_params["v0"]:.4f}', f'{heston_params["kappa"]:.4f}',
            f'{heston_params["theta"]:.4f}', f'{heston_params["sigma_v"]:.4f}',
            f'{heston_params["rho"]:.4f}',
            f'{half_life:.0f}', f'{heston_params["rmse"]:.6f}',
            '─── Volatility Analysis ───',
            f'{strategy_info.get("atm_iv", 0):.2%}', f'{rv_current:.2%}',
            f'{vol_premium:+.2%}',
            '─── Strategy P&L (MC) ───',
            f'${abs(net_premium):.2f} ({credit_debit})',
            f'${mean_pnl:.2f}', f'${median_pnl:.2f}', f'${std_pnl:.2f}',
            f'{win_rate:.1%}', f'${max_gain:.2f}', f'${max_loss:.2f}',
            f'${var_95:.2f}', f'${cvar_95:.2f}',
            f'{sharpe:.3f}'
        ]
    })

    st.dataframe(summary, use_container_width=True, hide_index=True, height=800)

    # =========================================================================
    # Variance path analysis (vol mean reversion under Heston)
    # =========================================================================
    st.markdown('<div class="section-header">7 │ Simulated Variance Paths — Mean Reversion in Action</div>',
                unsafe_allow_html=True)

    fig6, ax6 = plt.subplots(figsize=(14, 4), facecolor='#0e1117')
    ax6.set_facecolor('#161b22')
    ax6.tick_params(colors='#8b949e')
    ax6.spines['bottom'].set_color('#30363d')
    ax6.spines['left'].set_color('#30363d')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)

    for i in range(min(30, n_mc_paths)):
        ax6.plot(t_axis, np.sqrt(v_paths[i, :]), color='#58a6ff', alpha=0.08, linewidth=0.5)

    ax6.plot(t_axis, np.sqrt(np.mean(v_paths, axis=0)), color='#f0883e', linewidth=2.5,
             label='Mean Vol Path')
    ax6.axhline(np.sqrt(heston_params['theta']), color='#3fb950', linewidth=2,
                linestyle='--', label=f'θ = {np.sqrt(heston_params["theta"]):.2%}')
    ax6.axhline(np.sqrt(heston_params['v0']), color='#d2a8ff', linewidth=1,
                linestyle=':', label=f'v₀ = {np.sqrt(heston_params["v0"]):.2%}')

    ax6.set_xlabel('Days', color='#c9d1d9')
    ax6.set_ylabel('Instantaneous Vol (√v)', color='#c9d1d9')
    ax6.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax6.set_title('Heston Variance Paths — Vol Reverts to θ', color='#c9d1d9', fontweight='bold')
    ax6.legend(fontsize=10, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    plt.tight_layout()
    st.pyplot(fig6)

    st.success("✅ Analysis complete.")

else:
    # Landing state
    st.markdown("""
    ### How it works

    This tool exploits the empirical fact that **implied volatility mean-reverts** — when IV is
    elevated relative to realized vol, it tends to fall back, and vice versa.

    **Pipeline:**
    1. Pull live options data via `yfinance` → build the IV smile/surface
    2. Compare IV to historical realized vol → identify rich/cheap vol regions
    3. Construct a spread strategy that is short rich vol, long cheap vol
    4. Calibrate the **Heston stochastic volatility model** to the live smile
    5. Monte Carlo simulate 10K+ paths → compute strategy P&L distribution

    The Heston model naturally encodes mean reversion through its **κ** (speed) and **θ** (long-run level)
    parameters, making it the ideal framework for this analysis.

    **👈 Configure settings in the sidebar and hit Run Analysis.**
    """)
