"""Black-scholes-Merton pricing for European options.

Supports a continuous dividend yield q. All functions accept floats or numpy arrays (vectorised).
"""

from typing import Literal

import numpy as np 
from scipy.optimize import brentq
from scipy.stats import norm

OptionType = Literal["call", "put"]

def _d1_d2(S, K, T, r, sigma, q=0.0):
    """return the d1 and d2 terms of the Black-Scholes formula."""
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def _check_type(option_type: str) -> None:
    if option_type not in ("call", "put"):
        raise ValueError(f"option-type must be 'call' or 'put', got {option_type!r}")


def price(S, K, T, r, sigma, q=0.0, option_type: OptionType = "call"):
    """Price of a European option.

    S: spot, K: strike, T: maturity in years, r: risk-free rate,
    sigma: volatility, q: continuous dividend yield.
    """

    _check_type(option_type)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    if option_type == "call":
        return S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    return K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)


def delta(S, K, T, r, sigma, q=0.0, option_type: OptionType = "call"):
    """Sensitivity of the price to the spot (dV/dS)"""
    _check_type(option_type)
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    df_q = np.exp(-q * T)
    if option_type == "call":
        return df_q * norm.cdf(d1)
    return df_q * (norm.cdf(d1) - 1.0)


def gamma(S, K, T, r, sigma, q=0.0):
    """Second derivative w.r.t. the spot (identical for calls and puts)"""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    """Sensitivity to volatility, for 1.00 (100%) move in sigma """
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, q=0.0, option_type: OptionType = "call"):
    """Sensitivity to the passage of time (dV/dt, per year)"""
    _check_type(option_type)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    decay = -S * df_q * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        return decay - r * K * df_r * norm.cdf(d2) + q * S * df_q * norm.cdf(d1)
    return decay +r * K * df_r * norm.cdf(-d2) - q * S * df_q * norm.cdf(-d1)


def rho(S, K, T, r, sigma, q=0.0, option_type: OptionType = "call"):
    """Sensitivity to the risl-free rate, for a 1.00 (100%) move in r"""
    _check_type(option_type)
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    df_r = np.exp(-r * T)
    if option_type == "call":
        return K * T * df_r *norm.cdf(d2)
    return - K * T * df_r * norm.cdf(-d2)


def implied_volatility(market_price, S, K, T, r, q=0.0, option_type: OptionType="call", low=1e-6, high=5.0):
    """Volatility that repriduces a market price (Brent's root finder)"""
    objective = lambda s: price(S, K, T, r, s, q, option_type) - market_price
    return brentq(objective, low, high, xtol=1e-10)