import pytest

from option_pricer import black_scholes as bs
from option_pricer import fast
from option_pricer import monte_carlo as mc

P = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)


def _cases():
    return [(m, t) for m in ("plain", "antithetic") for t in ("call", "put")]


@pytest.mark.parametrize("case", _cases())
def test_cpp_converges_to_black_scholes(case):
    method, option_type = case
    res = fast.price_european(**P, q=0.01, option_type=option_type,
                              n_paths=500_000, method=method, seed=42)
    exact = bs.price(**P, q=0.01, option_type=option_type)
    assert abs(res.price - exact) < 4 * res.std_error


def test_cpp_antithetic_lowers_std_error():
    plain = fast.price_european(**P, n_paths=200_000, method="plain", seed=1)
    anti = fast.price_european(**P, n_paths=200_000, method="antithetic", seed=1)
    assert anti.std_error < plain.std_error


def test_cpp_is_reproducible():
    a = fast.price_european(**P, n_paths=100_000, seed=7)
    b = fast.price_european(**P, n_paths=100_000, seed=7)
    assert a.price == b.price


def test_cpp_asian_agrees_with_numpy():
    cpp = fast.price_asian_arithmetic(**P, n_steps=52, n_paths=100_000, seed=0)
    py = mc.price_asian_arithmetic(**P, n_steps=52, n_paths=100_000, seed=0)
    # Independent random streams: compare within combined standard errors
    combined_se = (cpp.std_error**2 + py.std_error**2) ** 0.5
    assert abs(cpp.price - py.price) < 4 * combined_se


def test_cpp_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        fast.price_european(S=100.0, K=-1.0, T=1.0, r=0.05, sigma=0.2)
    with pytest.raises(ValueError):
        fast.price_european(**P, option_type="straddle")
