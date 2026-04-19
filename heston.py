"""
Heston Stochastic Volatility Model
- Closed-form characteristic function pricing
- Calibration to market implied vols
- Monte Carlo simulation for strategy P&L
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm


# =============================================================================
# Black-Scholes helpers
# =============================================================================

def bs_price(S, K, T, r, sigma, option_type="call"):
    """Standard Black-Scholes price."""
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return max(S - K, 0)
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_vega(S, K, T, r, sigma):
    """Black-Scholes vega."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def implied_vol_from_price(price, S, K, T, r, option_type="call", tol=1e-6, max_iter=100):
    """Newton-Raphson implied vol inversion."""
    if T <= 0:
        return np.nan
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    if price <= intrinsic + 1e-10:
        return np.nan

    sigma = 0.25  # initial guess
    for _ in range(max_iter):
        p = bs_price(S, K, T, r, sigma, option_type)
        v = bs_vega(S, K, T, r, sigma)
        if v < 1e-12:
            break
        sigma_new = sigma - (p - price) / v
        if sigma_new <= 0.001:
            sigma_new = 0.001
        if abs(sigma_new - sigma) < tol:
            return sigma_new
        sigma = sigma_new
    return sigma if 0.01 < sigma < 3.0 else np.nan


# =============================================================================
# Heston characteristic function & pricing
# =============================================================================

def heston_char_func(u, S, K, T, r, v0, kappa, theta, sigma_v, rho):
    """Heston model characteristic function (Albrecher et al. formulation)."""
    i = complex(0, 1)
    d = np.sqrt((rho * sigma_v * i * u - kappa)**2 + sigma_v**2 * (i * u + u**2))
    g = (kappa - rho * sigma_v * i * u - d) / (kappa - rho * sigma_v * i * u + d)

    C = (kappa * theta / sigma_v**2) * (
        (kappa - rho * sigma_v * i * u - d) * T
        - 2.0 * np.log((1.0 - g * np.exp(-d * T)) / (1.0 - g))
    )
    D = ((kappa - rho * sigma_v * i * u - d) / sigma_v**2) * (
        (1.0 - np.exp(-d * T)) / (1.0 - g * np.exp(-d * T))
    )

    return np.exp(C + D * v0 + i * u * np.log(S * np.exp(r * T)))


def heston_call_price(S, K, T, r, v0, kappa, theta, sigma_v, rho, N=200):
    """Heston call price via numerical integration of the characteristic function."""
    du = 0.5
    integral1 = 0.0
    integral2 = 0.0
    i = complex(0, 1)

    for j in range(1, N + 1):
        u = j * du
        # P1 integrand
        f1 = heston_char_func(u - i, S, K, T, r, v0, kappa, theta, sigma_v, rho)
        f1 /= (heston_char_func(-i, S, K, T, r, v0, kappa, theta, sigma_v, rho))
        integrand1 = np.real(np.exp(-i * u * np.log(K)) * f1 / (i * u))
        integral1 += integrand1 * du

        # P2 integrand
        f2 = heston_char_func(u, S, K, T, r, v0, kappa, theta, sigma_v, rho)
        integrand2 = np.real(np.exp(-i * u * np.log(K)) * f2 / (i * u))
        integral2 += integrand2 * du

    P1 = 0.5 + integral1 / np.pi
    P2 = 0.5 + integral2 / np.pi

    return S * P1 - K * np.exp(-r * T) * P2


def heston_implied_vol(S, K, T, r, v0, kappa, theta, sigma_v, rho):
    """Get BS implied vol from a Heston price."""
    try:
        price = heston_call_price(S, K, T, r, v0, kappa, theta, sigma_v, rho)
        if price <= 0 or np.isnan(price):
            return np.nan
        iv = implied_vol_from_price(price, S, K, T, r, "call")
        return iv
    except:
        return np.nan


# =============================================================================
# Calibration
# =============================================================================

def calibrate_heston(market_data, S, r):
    """
    Calibrate Heston to market implied vols.

    market_data: list of dicts with keys 'K', 'T', 'iv' (market implied vol)
    Returns: dict with v0, kappa, theta, sigma_v, rho
    """
    strikes = np.array([d['K'] for d in market_data])
    expiries = np.array([d['T'] for d in market_data])
    market_ivs = np.array([d['iv'] for d in market_data])

    # Weight ATM options more heavily
    moneyness = strikes / S
    weights = np.exp(-5.0 * (moneyness - 1.0)**2)
    weights /= weights.sum()

    def objective(params):
        v0, kappa, theta, sigma_v, rho = params
        total_error = 0.0
        for idx in range(len(market_data)):
            model_iv = heston_implied_vol(S, strikes[idx], expiries[idx], r,
                                          v0, kappa, theta, sigma_v, rho)
            if np.isnan(model_iv):
                total_error += 1.0 * weights[idx]
            else:
                total_error += weights[idx] * (model_iv - market_ivs[idx])**2
        return total_error

    bounds = [
        (0.005, 1.0),    # v0
        (0.1, 10.0),     # kappa (mean reversion speed)
        (0.005, 1.0),    # theta (long-run variance)
        (0.05, 2.0),     # sigma_v (vol of vol)
        (-0.99, -0.01),  # rho (correlation, negative for equities)
    ]

    # Global search first
    result_de = differential_evolution(objective, bounds, seed=42,
                                       maxiter=100, tol=1e-8, popsize=20)

    # Polish with local optimizer
    result = minimize(objective, result_de.x, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-8})

    v0, kappa, theta, sigma_v, rho = result.x
    return {
        'v0': v0, 'kappa': kappa, 'theta': theta,
        'sigma_v': sigma_v, 'rho': rho,
        'rmse': np.sqrt(result.fun / len(market_data))
    }


# =============================================================================
# Monte Carlo Simulation
# =============================================================================

def simulate_heston_paths(S0, v0, kappa, theta, sigma_v, rho, r, T,
                          n_paths=10000, n_steps=252):
    """
    Simulate Heston paths using QE (Quadratic Exponential) scheme
    for variance and Euler for log-price.
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    v[:, 0] = v0

    for t in range(n_steps):
        # Correlated Brownian motions
        Z1 = np.random.standard_normal(n_paths)
        Z2 = np.random.standard_normal(n_paths)
        Wv = Z1
        Ws = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        # Variance process (full truncation scheme)
        v_pos = np.maximum(v[:, t], 0)
        v[:, t + 1] = (v[:, t]
                       + kappa * (theta - v_pos) * dt
                       + sigma_v * np.sqrt(v_pos) * sqrt_dt * Wv)
        v[:, t + 1] = np.maximum(v[:, t + 1], 0)  # absorbing at zero

        # Log-price process
        S[:, t + 1] = S[:, t] * np.exp(
            (r - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * Ws
        )

    return S, v


def price_strategy_mc(S_paths, v_paths, strategy_legs, r, T):
    """
    Price a multi-leg options strategy using Monte Carlo terminal values.

    strategy_legs: list of dicts with:
        'type': 'call' or 'put'
        'K': strike
        'position': +1 (long) or -1 (short)
        'premium': entry premium paid/received (positive = paid)
    """
    n_paths = S_paths.shape[0]
    S_T = S_paths[:, -1]

    pnl = np.zeros(n_paths)

    for leg in strategy_legs:
        K = leg['K']
        pos = leg['position']
        prem = leg['premium']

        if leg['type'] == 'call':
            payoff = np.maximum(S_T - K, 0)
        else:
            payoff = np.maximum(K - S_T, 0)

        # P&L = position * (payoff - premium_paid)
        # If short: position=-1, premium is received (negative cost)
        pnl += pos * (payoff * np.exp(-r * T) - prem)

    return pnl
