"""
Volatility Mean-Reversion Options Strategy — CLI
================================================
1. Pull SPY options data + VIX/price history via yfinance
2. Build implied vol smile/surface
3. Identify rich/cheap vol regions, construct mean-reversion strategy
4. Calibrate Heston model to the live smile
5. Monte Carlo simulate strategy P&L under Heston dynamics

Plots and a results CSV are written to the output directory.
"""

import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.interpolate import griddata

from heston import (
    calibrate_heston,
    heston_implied_vol,
    implied_vol_from_price,
    price_strategy_mc,
    simulate_heston_paths,
)

warnings.filterwarnings("ignore")

STRATEGIES = ["vega-neutral", "short-straddle", "put-ratio"]
STRATEGY_LABELS = {
    "vega-neutral": "Vega-Neutral Spread",
    "short-straddle": "Short Straddle + Wing Hedge",
    "put-ratio": "Put Ratio Spread",
}

BG, PANEL, GRID = "#0e1117", "#161b22", "#30363d"
TEXT, MUTED = "#c9d1d9", "#8b949e"
BLUE, ORANGE, GREEN, RED, PURPLE, YELLOW = (
    "#58a6ff", "#f0883e", "#3fb950", "#f85149", "#d2a8ff", "#e3b341",
)
EXPIRY_COLORS = [BLUE, ORANGE, PURPLE, GREEN, RED, YELLOW]


def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["left"].set_color(GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def save_fig(fig, path):
    fig.savefig(path, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def load_market_data(ticker, lookback_days):
    stock = yf.Ticker(ticker)
    end = datetime.now()
    start = end - timedelta(days=int(lookback_days * 1.5))
    hist = stock.history(start=start, end=end)

    vix_hist = yf.Ticker("^VIX").history(start=start, end=end)

    try:
        expiry_dates = list(stock.options[:6])
    except Exception:
        expiry_dates = []

    chains = {}
    for exp in expiry_dates:
        try:
            ch = stock.option_chain(exp)
            chains[exp] = {"calls": ch.calls, "puts": ch.puts}
        except Exception:
            continue

    if hist.empty:
        raise RuntimeError("No historical price data retrieved.")

    current_price = float(hist["Close"].iloc[-1])

    try:
        irx = yf.Ticker("^IRX").history(period="5d")
        risk_free = float(irx["Close"].iloc[-1]) / 100.0
    except Exception:
        risk_free = 0.045

    return hist, vix_hist, chains, expiry_dates, current_price, risk_free


def compute_realized_vol(prices, window):
    log_ret = np.log(prices / prices.shift(1)).dropna()
    return log_ret.rolling(window=window).std() * np.sqrt(252)


def build_vol_smile(chain_calls, chain_puts, S, T, r):
    rows = []
    for frame, opt_type in ((chain_calls, "call"), (chain_puts, "put")):
        for _, row in frame.iterrows():
            K = row["strike"]
            bid, ask = row.get("bid", 0) or 0, row.get("ask", 0) or 0
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else row.get("lastPrice", 0)
            if mid <= 0 or T <= 0:
                continue
            iv = row.get("impliedVolatility", None)
            if iv is None or iv <= 0 or iv > 3.0:
                iv = implied_vol_from_price(mid, S, K, T, r, opt_type)
            if iv is not None and not np.isnan(iv) and 0.01 < iv < 3.0:
                rows.append({
                    "K": K, "T": T, "iv": iv, "type": opt_type,
                    "moneyness": K / S, "mid_price": mid,
                })
    return rows


def construct_strategy(smile_df, S, strategy, vol_threshold, rv_mean):
    if smile_df.empty:
        return [], {}

    smile_df = smile_df.copy()
    smile_df["abs_moneyness"] = (smile_df["moneyness"] - 1.0).abs()
    atm_row = smile_df.loc[smile_df["abs_moneyness"].idxmin()]
    atm_iv = atm_row["iv"]
    iv_std = smile_df["iv"].std() or 1e-9
    smile_df["vol_richness"] = (smile_df["iv"] - rv_mean) / iv_std

    rich = smile_df[smile_df["vol_richness"] > vol_threshold]
    cheap = smile_df[smile_df["vol_richness"] < -vol_threshold * 0.5]

    info = {
        "atm_iv": atm_iv,
        "rv_mean": rv_mean,
        "rich_strikes": rich["K"].tolist(),
        "cheap_strikes": cheap["K"].tolist(),
    }

    legs = []
    calls = smile_df[smile_df["type"] == "call"]
    puts = smile_df[smile_df["type"] == "put"]

    if strategy == "vega-neutral":
        p = puts.sort_values("vol_richness", ascending=False)
        c = calls.sort_values("vol_richness", ascending=False)
        if len(p) >= 2 and len(c) >= 2:
            sp, sc, lp, lc = p.iloc[0], c.iloc[0], p.iloc[-1], c.iloc[-1]
            if lp["K"] < sp["K"] and lc["K"] > sc["K"]:
                legs = [
                    {"type": "put", "K": lp["K"], "position": 1,
                     "premium": lp["mid_price"], "iv": lp["iv"]},
                    {"type": "put", "K": sp["K"], "position": -1,
                     "premium": sp["mid_price"], "iv": sp["iv"]},
                    {"type": "call", "K": sc["K"], "position": -1,
                     "premium": sc["mid_price"], "iv": sc["iv"]},
                    {"type": "call", "K": lc["K"], "position": 1,
                     "premium": lc["mid_price"], "iv": lc["iv"]},
                ]

    elif strategy == "short-straddle":
        atm_c = calls.loc[calls["abs_moneyness"].idxmin()] if not calls.empty else None
        atm_p = puts.loc[puts["abs_moneyness"].idxmin()] if not puts.empty else None
        otm_c = calls[calls["moneyness"] > 1.04].sort_values("moneyness")
        otm_p = puts[puts["moneyness"] < 0.96].sort_values("moneyness", ascending=False)
        if atm_c is not None and atm_p is not None and not otm_c.empty and not otm_p.empty:
            wc, wp = otm_c.iloc[0], otm_p.iloc[0]
            legs = [
                {"type": "put", "K": atm_p["K"], "position": -1,
                 "premium": atm_p["mid_price"], "iv": atm_p["iv"]},
                {"type": "call", "K": atm_c["K"], "position": -1,
                 "premium": atm_c["mid_price"], "iv": atm_c["iv"]},
                {"type": "put", "K": wp["K"], "position": 1,
                 "premium": wp["mid_price"], "iv": wp["iv"]},
                {"type": "call", "K": wc["K"], "position": 1,
                 "premium": wc["mid_price"], "iv": wc["iv"]},
            ]

    elif strategy == "put-ratio":
        if not puts.empty:
            atm_p = puts.loc[puts["abs_moneyness"].idxmin()]
            otm_rich = puts[puts["moneyness"] < 0.96].sort_values(
                "vol_richness", ascending=False
            )
            if not otm_rich.empty:
                rich_otm = otm_rich.iloc[0]
                legs = [
                    {"type": "put", "K": atm_p["K"], "position": 1,
                     "premium": atm_p["mid_price"], "iv": atm_p["iv"]},
                    {"type": "put", "K": rich_otm["K"], "position": -2,
                     "premium": rich_otm["mid_price"], "iv": rich_otm["iv"]},
                ]

    if not legs and len(calls) >= 2 and len(puts) >= 2:
        c_sorted = calls.sort_values("K")
        p_sorted = puts.sort_values("K")
        atm_idx_c = (c_sorted["moneyness"] - 1.0).abs().idxmin()
        atm_idx_p = (p_sorted["moneyness"] - 1.0).abs().idxmin()
        legs = [
            {"type": "put", "K": p_sorted["K"].iloc[0], "position": 1,
             "premium": p_sorted["mid_price"].iloc[0], "iv": p_sorted["iv"].iloc[0]},
            {"type": "put", "K": p_sorted.loc[atm_idx_p, "K"], "position": -1,
             "premium": p_sorted.loc[atm_idx_p, "mid_price"],
             "iv": p_sorted.loc[atm_idx_p, "iv"]},
            {"type": "call", "K": c_sorted.loc[atm_idx_c, "K"], "position": -1,
             "premium": c_sorted.loc[atm_idx_c, "mid_price"],
             "iv": c_sorted.loc[atm_idx_c, "iv"]},
            {"type": "call", "K": c_sorted["K"].iloc[-1], "position": 1,
             "premium": c_sorted["mid_price"].iloc[-1], "iv": c_sorted["iv"].iloc[-1]},
        ]

    return legs, info


def plot_vol_history(rv, vix_series, rv_window, rv_mean, rv_std, path):
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(14, 8), height_ratios=[2, 1], facecolor=BG
    )
    for ax in (ax_a, ax_b):
        style_ax(ax)

    rv_plot = rv.dropna()
    ax_a.plot(rv_plot.index, rv_plot.values, color=BLUE, linewidth=1.5,
              label=f"{rv_window}d Realized Vol", alpha=0.9)

    vix_plot = vix_series.dropna()
    if not vix_plot.empty:
        ax_a.plot(vix_plot.index, vix_plot.values, color=ORANGE, linewidth=1.5,
                  label="VIX (Implied Vol)", alpha=0.9)

    ax_a.axhline(rv_mean, color=GREEN, linestyle="--", alpha=0.7,
                 label=f"RV Mean ({rv_mean:.1%})")
    ax_a.fill_between(rv_plot.index, rv_mean - rv_std, rv_mean + rv_std,
                      alpha=0.1, color=GREEN, label="±1σ Band")
    ax_a.fill_between(rv_plot.index, rv_mean - 2 * rv_std, rv_mean + 2 * rv_std,
                      alpha=0.05, color=GREEN, label="±2σ Band")
    ax_a.set_ylabel("Annualized Volatility", fontsize=11)
    ax_a.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax_a.legend(loc="upper right", fontsize=9, facecolor=PANEL,
                edgecolor=GRID, labelcolor=TEXT)
    ax_a.set_title("Implied vs Realized Volatility — Mean Reversion",
                   fontsize=14, fontweight="bold", pad=12)

    if not vix_plot.empty:
        common = rv_plot.index.intersection(vix_plot.index)
        if len(common) > 0:
            spread = vix_plot.loc[common] - rv_plot.loc[common]
            colors = [RED if x > 0 else GREEN for x in spread]
            ax_b.bar(common, spread, color=colors, alpha=0.6, width=1.5)
            ax_b.axhline(0, color=MUTED, linewidth=0.5)
            ax_b.set_ylabel("IV − RV Spread", fontsize=11)
            ax_b.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    fig.tight_layout()
    save_fig(fig, path)


def plot_smile_surface(smile_df_all, expiry_smiles, rv_mean, path):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    for ax in (ax_a, ax_b):
        style_ax(ax)

    for i, (exp_str, df_exp) in enumerate(expiry_smiles.items()):
        df_plot = df_exp[(df_exp["moneyness"] > 0.88) & (df_exp["moneyness"] < 1.12)]
        if df_plot.empty:
            continue
        df_avg = (df_plot.sort_values("moneyness")
                         .groupby("moneyness", as_index=False)["iv"].mean())
        ax_a.plot(df_avg["moneyness"], df_avg["iv"],
                  color=EXPIRY_COLORS[i % len(EXPIRY_COLORS)],
                  linewidth=2, marker="o", markersize=3, label=exp_str, alpha=0.85)

    ax_a.axvline(1.0, color=MUTED, linestyle=":", alpha=0.5, label="ATM")
    ax_a.axhline(rv_mean, color=GREEN, linestyle="--", alpha=0.5, label="RV Mean")
    ax_a.set_xlabel("Moneyness (K/S)")
    ax_a.set_ylabel("Implied Volatility")
    ax_a.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax_a.set_title("Vol Smile by Expiry", fontweight="bold")
    ax_a.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    if len(smile_df_all) > 20:
        m_grid = np.linspace(0.88, 1.12, 50)
        t_grid = np.linspace(smile_df_all["T"].min(), smile_df_all["T"].max(), 50)
        M, T = np.meshgrid(m_grid, t_grid)
        try:
            iv_surface = griddata(
                (smile_df_all["moneyness"].values, smile_df_all["T"].values),
                smile_df_all["iv"].values, (M, T), method="cubic",
            )
            c = ax_b.pcolormesh(M, T * 365, iv_surface, cmap="magma", shading="auto")
            cb = fig.colorbar(c, ax=ax_b, label="IV",
                              format=mticker.PercentFormatter(1.0))
            cb.ax.yaxis.label.set_color(TEXT)
            cb.ax.tick_params(colors=MUTED)
            ax_b.set_xlabel("Moneyness (K/S)")
            ax_b.set_ylabel("Days to Expiry")
            ax_b.set_title("Vol Surface", fontweight="bold")
        except Exception:
            ax_b.text(0.5, 0.5, "Insufficient data\nfor surface interpolation",
                      ha="center", va="center", color=MUTED, fontsize=12,
                      transform=ax_b.transAxes)

    fig.tight_layout()
    save_fig(fig, path)


def plot_payoff(legs, S, ticker, path):
    fig, ax = plt.subplots(figsize=(14, 4), facecolor=BG)
    style_ax(ax)

    S_range = np.linspace(S * 0.85, S * 1.15, 500)
    payoff = np.zeros_like(S_range)
    for leg in legs:
        if leg["type"] == "call":
            payoff += leg["position"] * (np.maximum(S_range - leg["K"], 0) - leg["premium"])
        else:
            payoff += leg["position"] * (np.maximum(leg["K"] - S_range, 0) - leg["premium"])

    ax.fill_between(S_range, payoff, 0, where=(payoff > 0), alpha=0.3, color=GREEN)
    ax.fill_between(S_range, payoff, 0, where=(payoff < 0), alpha=0.3, color=RED)
    ax.plot(S_range, payoff, color="#e2e8f0", linewidth=2)
    ax.axhline(0, color=MUTED, linewidth=0.5)
    ax.axvline(S, color=ORANGE, linestyle=":", alpha=0.7, label=f"Spot (${S:.0f})")
    for leg in legs:
        ax.axvline(leg["K"], color=BLUE, linestyle="--", alpha=0.3)

    ax.set_xlabel(f"{ticker} Price at Expiry")
    ax.set_ylabel("P&L per Contract ($)")
    ax.set_title("Strategy Payoff at Expiry", fontweight="bold")
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    fig.tight_layout()
    save_fig(fig, path)


def plot_heston_fit(unique_calib, heston_params, S, r, path):
    fig, ax = plt.subplots(figsize=(14, 4), facecolor=BG)
    style_ax(ax)

    unique_T = sorted({round(d["T"], 4) for d in unique_calib})
    for i, T_val in enumerate(unique_T[:4]):
        this_exp = [d for d in unique_calib if abs(d["T"] - T_val) < 0.005]
        if len(this_exp) < 3:
            continue
        this_exp.sort(key=lambda x: x["K"])
        market_K = [d["K"] / S for d in this_exp]
        market_iv = [d["iv"] for d in this_exp]
        model_iv = [heston_implied_vol(S, d["K"], d["T"], r,
                                       heston_params["v0"], heston_params["kappa"],
                                       heston_params["theta"], heston_params["sigma_v"],
                                       heston_params["rho"]) for d in this_exp]
        c = EXPIRY_COLORS[i % len(EXPIRY_COLORS)]
        dte = int(T_val * 365)
        ax.scatter(market_K, market_iv, color=c, s=25, alpha=0.7, zorder=5,
                   label=f"Market ({dte}d)")
        valid = [(k, m) for k, m in zip(market_K, model_iv)
                 if m is not None and not np.isnan(m)]
        if valid:
            ax.plot([v[0] for v in valid], [v[1] for v in valid],
                    color=c, linewidth=2, alpha=0.8, linestyle="--")

    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Implied Volatility")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Heston Fit vs Market (dots = market, dashes = model)",
                 fontweight="bold")
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    fig.tight_layout()
    save_fig(fig, path)


def plot_pnl_and_paths(pnl, S_paths, legs, target_T, n_steps, ticker,
                       mean_pnl, var_95, path):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    for ax in (ax_a, ax_b):
        style_ax(ax)

    counts, bins, patches = ax_a.hist(pnl, bins=80, alpha=0.8, edgecolor="none")
    for patch, b in zip(patches, bins[:-1]):
        patch.set_facecolor(GREEN if b >= 0 else RED)
    ax_a.axvline(mean_pnl, color=ORANGE, linewidth=2, linestyle="-",
                 label=f"Mean ${mean_pnl:.2f}")
    ax_a.axvline(var_95, color=PURPLE, linewidth=1.5, linestyle="--",
                 label=f"5% VaR ${var_95:.2f}")
    ax_a.axvline(0, color="#e2e8f0", linewidth=0.8, linestyle=":")
    ax_a.set_xlabel("P&L ($)")
    ax_a.set_ylabel("Frequency")
    ax_a.set_title("Strategy P&L Distribution", fontweight="bold")
    ax_a.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    t_axis = np.linspace(0, target_T * 365, n_steps + 1)
    n_show = min(50, S_paths.shape[0])
    for i in range(n_show):
        ax_b.plot(t_axis, S_paths[i, :], color=BLUE, alpha=0.08, linewidth=0.5)
    ax_b.plot(t_axis, np.mean(S_paths, axis=0), color=ORANGE, linewidth=2,
              label="Mean Path")
    ax_b.plot(t_axis, np.percentile(S_paths, 5, axis=0), color=RED,
              linewidth=1, linestyle="--", label="5th Percentile")
    ax_b.plot(t_axis, np.percentile(S_paths, 95, axis=0), color=GREEN,
              linewidth=1, linestyle="--", label="95th Percentile")
    for leg in legs:
        ax_b.axhline(leg["K"], color=PURPLE, linestyle=":", alpha=0.4)
    ax_b.set_xlabel("Days")
    ax_b.set_ylabel(f"{ticker} Price")
    ax_b.set_title("Simulated Price Paths (Heston)", fontweight="bold")
    ax_b.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    fig.tight_layout()
    save_fig(fig, path)


def plot_variance_paths(v_paths, heston_params, target_T, n_steps, path):
    fig, ax = plt.subplots(figsize=(14, 4), facecolor=BG)
    style_ax(ax)

    t_axis = np.linspace(0, target_T * 365, n_steps + 1)
    for i in range(min(30, v_paths.shape[0])):
        ax.plot(t_axis, np.sqrt(v_paths[i, :]), color=BLUE, alpha=0.08, linewidth=0.5)

    ax.plot(t_axis, np.sqrt(np.mean(v_paths, axis=0)), color=ORANGE, linewidth=2.5,
            label="Mean Vol Path")
    ax.axhline(np.sqrt(heston_params["theta"]), color=GREEN, linewidth=2,
               linestyle="--", label=f"θ = {np.sqrt(heston_params['theta']):.2%}")
    ax.axhline(np.sqrt(heston_params["v0"]), color=PURPLE, linewidth=1,
               linestyle=":", label=f"v₀ = {np.sqrt(heston_params['v0']):.2%}")
    ax.set_xlabel("Days")
    ax.set_ylabel("Instantaneous Vol (√v)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Heston Variance Paths — Vol Reverts to θ", fontweight="bold")
    ax.legend(fontsize=10, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    fig.tight_layout()
    save_fig(fig, path)


def run_analysis(args):
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    strategy_label = STRATEGY_LABELS[args.strategy]

    print(f"\n[1/7] Fetching market data for {args.ticker}...")
    hist, vix_hist, chains, expiry_dates, S, r = load_market_data(
        args.ticker, args.lookback_days
    )
    if not chains:
        raise RuntimeError("No options data available for this ticker.")

    print(f"      spot=${S:.2f}  risk-free={r:.2%}  expiries={len(expiry_dates)}")

    print(f"\n[2/7] Computing realized vol ({args.rv_window}d window)...")
    rv = compute_realized_vol(hist["Close"], args.rv_window)
    rv_clean = rv.dropna()
    rv_mean = float(rv_clean.mean())
    rv_std = float(rv_clean.std())
    rv_current = float(rv_clean.iloc[-1])

    vix_series = vix_hist["Close"].reindex(hist.index, method="ffill") / 100.0
    vix_clean = vix_series.dropna()
    vix_current = float(vix_clean.iloc[-1]) if not vix_clean.empty else 0.0
    vol_premium = vix_current - rv_current

    print(f"      RV={rv_current:.2%}  VIX={vix_current:.2%}  "
          f"premium={vol_premium:+.2%}")

    plot_vol_history(rv, vix_series, args.rv_window, rv_mean, rv_std,
                     os.path.join(out_dir, "01_vol_history.png"))

    print("\n[3/7] Building vol smile / surface...")
    all_smile = []
    expiry_smiles = {}
    for exp_str in expiry_dates:
        if exp_str not in chains:
            continue
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
        T = max((exp_date - datetime.now()).days / 365.0, 0.001)
        if T < 0.01 or T > 2.0:
            continue
        smile = build_vol_smile(chains[exp_str]["calls"], chains[exp_str]["puts"], S, T, r)
        if smile:
            all_smile.extend(smile)
            expiry_smiles[exp_str] = pd.DataFrame(smile)

    if not all_smile:
        raise RuntimeError("Could not build vol smile from options data.")

    smile_df_all = pd.DataFrame(all_smile)
    smile_df_all = smile_df_all[
        (smile_df_all["moneyness"] > 0.85) & (smile_df_all["moneyness"] < 1.15)
    ].copy()

    print(f"      {len(smile_df_all)} vol points across {len(expiry_smiles)} expiries")
    plot_smile_surface(smile_df_all, expiry_smiles, rv_mean,
                       os.path.join(out_dir, "02_smile_surface.png"))

    print(f"\n[4/7] Constructing strategy: {strategy_label}...")
    target_exp, target_smile_df, target_T = None, None, None
    for exp_str in expiry_dates:
        if exp_str in expiry_smiles and len(expiry_smiles[exp_str]) >= 10:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            T_cand = (exp_date - datetime.now()).days / 365.0
            if 0.03 < T_cand < 0.25:
                target_exp = exp_str
                target_smile_df = expiry_smiles[exp_str].copy()
                target_T = T_cand
                break

    if target_smile_df is None:
        for exp_str in expiry_dates:
            if exp_str in expiry_smiles and len(expiry_smiles[exp_str]) >= 6:
                target_exp = exp_str
                target_smile_df = expiry_smiles[exp_str].copy()
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
                target_T = max((exp_date - datetime.now()).days / 365.0, 0.01)
                break

    if target_smile_df is None:
        raise RuntimeError("Not enough options data to construct strategy.")

    target_smile_df = target_smile_df[
        (target_smile_df["moneyness"] > 0.90) & (target_smile_df["moneyness"] < 1.10)
    ].copy()

    legs, strategy_info = construct_strategy(
        target_smile_df, S, args.strategy, args.vol_threshold, rv_mean
    )
    if not legs:
        raise RuntimeError(
            "Could not construct strategy with current data. "
            "Try adjusting --vol-threshold."
        )

    print(f"      target expiry={target_exp} ({int(target_T*365)} DTE)  legs={len(legs)}")

    leg_rows, net_premium = [], 0.0
    for leg in legs:
        pos_str = f"{'Long' if leg['position'] > 0 else 'Short'} {abs(leg['position'])}"
        leg_rows.append({
            "Position": pos_str,
            "Type": leg["type"].upper(),
            "Strike": f"${leg['K']:.0f}",
            "Premium": f"${leg['premium']:.2f}",
            "IV": f"{leg['iv']:.1%}",
            "Moneyness": f"{leg['K']/S:.3f}",
        })
        net_premium += leg["position"] * leg["premium"]
    legs_df = pd.DataFrame(leg_rows)
    credit_debit = "CREDIT" if net_premium < 0 else "DEBIT"

    print("\n  Strategy Legs:")
    print(legs_df.to_string(index=False))
    print(f"  Net Premium: ${abs(net_premium):.2f} ({credit_debit})")
    legs_df.to_csv(os.path.join(out_dir, "strategy_legs.csv"), index=False)
    plot_payoff(legs, S, args.ticker, os.path.join(out_dir, "03_strategy_payoff.png"))

    print("\n[5/7] Calibrating Heston model (this may take 30-60s)...")
    calib_data = [
        {"K": row["K"], "T": row["T"], "iv": row["iv"]}
        for _, row in smile_df_all.iterrows()
        if row["type"] == "call" and 0.90 < row["moneyness"] < 1.10
    ]
    if len(calib_data) < 5:
        calib_data = [
            {"K": row["K"], "T": row["T"], "iv": row["iv"]}
            for _, row in smile_df_all.iterrows()
            if 0.90 < row["moneyness"] < 1.10
        ]

    seen, unique_calib = set(), []
    for d in calib_data:
        key = (round(d["K"], 1), round(d["T"], 4))
        if key not in seen:
            seen.add(key)
            unique_calib.append(d)

    if len(unique_calib) < 5:
        raise RuntimeError("Not enough data points for Heston calibration.")

    heston_params = calibrate_heston(unique_calib, S, r)
    kappa = heston_params["kappa"]
    half_life = np.log(2) / kappa * 252

    print(f"      v0={heston_params['v0']:.4f}  kappa={kappa:.2f}  "
          f"theta={heston_params['theta']:.4f}")
    print(f"      sigma_v={heston_params['sigma_v']:.4f}  "
          f"rho={heston_params['rho']:.4f}  RMSE={heston_params['rmse']:.4f}")
    print(f"      MR half-life: {half_life:.0f} days ({half_life/21:.1f} months)")

    plot_heston_fit(unique_calib, heston_params, S, r,
                    os.path.join(out_dir, "04_heston_fit.png"))

    print(f"\n[6/7] Monte Carlo simulation ({args.mc_paths:,} paths)...")
    np.random.seed(args.seed)
    n_steps = max(int(target_T * 252), 20)
    S_paths, v_paths = simulate_heston_paths(
        S, heston_params["v0"], heston_params["kappa"], heston_params["theta"],
        heston_params["sigma_v"], heston_params["rho"], r, target_T,
        n_paths=args.mc_paths, n_steps=n_steps,
    )
    pnl = price_strategy_mc(S_paths, v_paths, legs, r, target_T)

    mean_pnl = float(np.mean(pnl))
    median_pnl = float(np.median(pnl))
    std_pnl = float(np.std(pnl))
    sharpe = (mean_pnl / std_pnl * np.sqrt(252 / (target_T * 252))
              if std_pnl > 0 else 0.0)
    win_rate = float(np.mean(pnl > 0))
    max_loss, max_gain = float(np.min(pnl)), float(np.max(pnl))
    var_95 = float(np.percentile(pnl, 5))
    cvar_95 = float(np.mean(pnl[pnl <= var_95]))

    print(f"      E[PnL]=${mean_pnl:.2f}  win_rate={win_rate:.1%}  "
          f"Sharpe={sharpe:.2f}  95% CVaR=${cvar_95:.2f}")

    plot_pnl_and_paths(pnl, S_paths, legs, target_T, n_steps, args.ticker,
                       mean_pnl, var_95,
                       os.path.join(out_dir, "05_pnl_and_paths.png"))
    plot_variance_paths(v_paths, heston_params, target_T, n_steps,
                        os.path.join(out_dir, "06_variance_paths.png"))

    print("\n[7/7] Writing results summary...")
    summary = pd.DataFrame({
        "Metric": [
            "Underlying", "Spot Price", "Risk-Free Rate", "Strategy", "Expiry",
            "DTE",
            "Heston v0", "Heston kappa", "Heston theta", "Heston sigma_v",
            "Heston rho", "MR Half-Life (days)", "Calibration RMSE",
            "ATM IV", f"{args.rv_window}d Realized Vol", "Vol Risk Premium",
            "Net Premium", "E[P&L]", "Median P&L", "Std Dev P&L",
            "Win Rate", "Max Gain", "Max Loss", "95% VaR", "95% CVaR",
            "Annualized Sharpe",
        ],
        "Value": [
            args.ticker, f"${S:.2f}", f"{r:.2%}", strategy_label, target_exp,
            f"{int(target_T*365)}",
            f"{heston_params['v0']:.4f}", f"{heston_params['kappa']:.4f}",
            f"{heston_params['theta']:.4f}", f"{heston_params['sigma_v']:.4f}",
            f"{heston_params['rho']:.4f}", f"{half_life:.0f}",
            f"{heston_params['rmse']:.6f}",
            f"{strategy_info.get('atm_iv', 0):.2%}", f"{rv_current:.2%}",
            f"{vol_premium:+.2%}",
            f"${abs(net_premium):.2f} ({credit_debit})",
            f"${mean_pnl:.2f}", f"${median_pnl:.2f}", f"${std_pnl:.2f}",
            f"{win_rate:.1%}", f"${max_gain:.2f}", f"${max_loss:.2f}",
            f"${var_95:.2f}", f"${cvar_95:.2f}",
            f"{sharpe:.3f}",
        ],
    })
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))
    print("=" * 60)
    print(f"\nAll outputs saved to: {out_dir}/")


def parse_args():
    p = argparse.ArgumentParser(
        description="Volatility mean-reversion options strategy engine (CLI).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ticker", default="SPY", help="Underlying ticker symbol.")
    p.add_argument("--lookback-days", type=int, default=252,
                   help="Historical price lookback in calendar days.")
    p.add_argument("--rv-window", type=int, default=30,
                   help="Realized-vol rolling window (trading days).")
    p.add_argument("--mc-paths", type=int, default=10000,
                   help="Number of Monte Carlo paths.")
    p.add_argument("--vol-threshold", type=float, default=1.5,
                   help="Std-devs above/below mean to flag rich/cheap vol.")
    p.add_argument("--strategy", choices=STRATEGIES, default="vega-neutral",
                   help="Strategy type.")
    p.add_argument("--output-dir", default="output",
                   help="Directory for plots and CSV results.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the Monte Carlo simulation.")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        run_analysis(args)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
