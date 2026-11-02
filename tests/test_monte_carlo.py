import numpy as np
import pytest

from option_pricer import black_scholes as bs
from option_pricer import monte_carlo as mc

P = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)
GBM = dict(S=100.0, T=1.0, r=0.05, sigma=0.20)  # no strike for paths
METHODS = ["plain", "antithetic", "control"]
TYPES = ["call", "put"]


def _cases():
    return [(m, t) for m in METHODS for t in TYPES]


@pytest.mark.parametrize("case", _cases())
def test_mc_converges_to_black_scholes(case):
    method, option_type = case
    res = mc.price_european(**P, q=0.01, option_type=option_type,
                            n_paths=200_000, method=method, seed=42)
    exact = bs.price(**P, q=0.01, option_type=option_type)
    # Within 4 standard errors: fails by chance with probability < 1e-4
    assert abs(res.price - exact) < 4 * res.std_error


@pytest.mark.parametrize("method", ["antithetic", "control"])
def test_variance_reduction_lowers_std_error(method):
    plain = mc.price_european(**P, n_paths=100_000, method="plain", seed=1)
    reduced = mc.price_european(**P, n_paths=100_000, method=method, seed=1)
    assert reduced.std_error < plain.std_error


def test_seed_makes_results_reproducible():
    a = mc.price_european(**P, n_paths=10_000, seed=7)
    b = mc.price_european(**P, n_paths=10_000, seed=7)
    assert a.price == b.price


def test_confidence_interval_contains_price():
    res = mc.price_european(**P, n_paths=10_000, seed=3)
    low, high = res.conf_int_95
    assert low < res.price < high


def test_paths_shape_and_start():
    paths = mc.simulate_paths(**GBM, n_steps=12, n_paths=500, seed=0)
    assert paths.shape == (500, 13)
    assert np.all(paths[:, 0] == P["S"])


def test_paths_martingale_under_risk_neutral_measure():
    # E[S_T] = S * exp(rT)
    paths = mc.simulate_paths(**GBM, n_steps=4, n_paths=200_000, seed=0)
    terminal = paths[:, -1]
    se = terminal.std(ddof=1) / np.sqrt(terminal.size)
    assert abs(terminal.mean() - P["S"] * np.exp(P["r"] * P["T"])) < 4 * se


def test_asian_call_cheaper_than_european_call():
    # Averaging reduces volatility, hence the option value
    asian = mc.price_asian_arithmetic(**P, n_steps=52, n_paths=50_000, seed=0)
    european = bs.price(**P, option_type="call")
    assert 0 < asian.price < european