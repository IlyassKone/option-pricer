"""Monte Carlo pricing under Black-Scholes dynamics (geometric Brownian motion).

Three estimators for European options:
- "plain":      standard Monte Carlo
- "antithetic": each draw Z is paired with -Z
- "control":    control variate on the discounted terminal spot,
                whose expectation S * exp(-qT) is known exactly
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from option_pricer.black_scholes import OptionType, _check_type

Method = Literal["plain", "antithetic", "control"]


@dataclass(frozen=True)
class MCResult:
    price: float
    std_error: float
    n_paths: int

    @property
    def conf_int_95(self) -> tuple[float, float]:
        half = 1.96 * self.std_error
        return self.price - half, self.price + half


def _payoff(ST, K, option_type: OptionType):
    if option_type == "call":
        return np.maximum(ST - K, 0.0)
    return np.maximum(K - ST, 0.0)


def _terminal_spot(S, T, r, sigma, q, z):
    """Exact GBM solution: S_T = S exp((r - q - sigma^2/2) T + sigma sqrt(T) Z)."""
    return S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def price_european(S, K, T, r, sigma, q=0.0, option_type: OptionType = "call",
                   n_paths: int = 100_000, method: Method = "plain",
                   seed: int | None = None) -> MCResult:
    """Monte Carlo price of a European option, with its standard error."""
    _check_type(option_type)
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)

    if method == "plain":
        z = rng.standard_normal(n_paths)
        samples = disc * _payoff(_terminal_spot(S, T, r, sigma, q, z), K, option_type)

    elif method == "antithetic":
        z = rng.standard_normal(n_paths // 2)
        pay_plus = _payoff(_terminal_spot(S, T, r, sigma, q, z), K, option_type)
        pay_minus = _payoff(_terminal_spot(S, T, r, sigma, q, -z), K, option_type)
        # Each pair average is one independent sample
        samples = disc * 0.5 * (pay_plus + pay_minus)

    elif method == "control":
        z = rng.standard_normal(n_paths)
        ST = _terminal_spot(S, T, r, sigma, q, z)
        y = disc * _payoff(ST, K, option_type)
        x = disc * ST                       # control variate
        x_mean = S * np.exp(-q * T)         # its exact expectation
        cov = np.cov(y, x)
        beta = cov[0, 1] / cov[1, 1]        # optimal coefficient
        samples = y - beta * (x - x_mean)

    else:
        raise ValueError(f"unknown method {method!r}")

    n = samples.size
    return MCResult(float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(n)), n)


def simulate_paths(S, T, r, sigma, q=0.0, n_steps: int = 252,
                   n_paths: int = 10_000, seed: int | None = None):
    """Simulate GBM paths on a uniform grid. Shape: (n_paths, n_steps + 1)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(log_increments, axis=1)
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])
    return S * np.exp(log_paths)


def price_asian_arithmetic(S, K, T, r, sigma, q=0.0,
                           option_type: OptionType = "call",
                           n_steps: int = 252, n_paths: int = 50_000,
                           seed: int | None = None) -> MCResult:
    """Arithmetic-average Asian option: no closed form, Monte Carlo is the tool."""
    _check_type(option_type)
    paths = simulate_paths(S, T, r, sigma, q, n_steps, n_paths, seed)
    avg = paths[:, 1:].mean(axis=1)
    samples = np.exp(-r * T) * _payoff(avg, K, option_type)
    return MCResult(float(samples.mean()),
                    float(samples.std(ddof=1) / np.sqrt(n_paths)), n_paths)