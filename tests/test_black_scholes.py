import numpy as np
import pytest

from option_pricer import black_scholes as bs 

# reference case: S=100, K=100, T=1, r= 5%, sigma=20%, no dividend
P = dict(S=100.0, K=100.0, T= 1.0, r=0.05, sigma=0.20)

def test_reference_call_price():
    # Known textbook value (Hull)
    assert bs.price(**P, option_type="call") == pytest.approx(10.4506, abs=1e-4)


def test_put_call_parity():
    q = 0.02
    call = bs.price(**P, q=q, option_type="call")
    put = bs.price(**P, q=q, option_type="put")
    S, K, T, r = P["S"], P["K"], P["T"], P["r"]
    assert call - put == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T))


    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_greeks_match_finite_differences(option_type):
        h = 1e-4

        def f(**kw):
            args = {**P, **kw}
            return bs.price(**args, option_type=option_type)

        
        S, sigma, r, T = P["S"], P["sigma"], P["r"], P["T"]
        fd_delta = (f(S=S + h) - f(S=S - h)) / (2 * h)
        fd_gamma = (f(S=S + h) - 2 * f() + f(S=S - h)) / h**2
        fd_vega = (f(sigma=sigma + h) - f(sigma=sigma - h)) / (2 * h)
        fd_rho = (f(r=r + h) - f(r=r - h)) / (2 * h)
        # theta = dV/dt = -dV/dT
        fd_theta = -(f(T=T + h) - f(T=T - h)) / (2 * h)

        assert bs.delta(**P, option_type=option_type) == pytest.approx(fd_delta, rel=1e-5)
        assert bs.gamma(**P) == pytest.approx(fd_gamma, rel=1e-3)
        assert bs.vega(**P) == pytest.approx(fd_vega, rel=1e-5)
        assert bs.rho(**P, option_type=option_type) == pytest.approx(fd_rho, rel=1e-5)
        assert bs.theta(**P, option_type=option_type) == pytest.approx(fd_theta, rel=1e-5)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_implied_volatility_round_trip(option_type):
    target = bs.price(**P, option_type=option_type)
    args = {k: v for k, v in P.items() if k != "sigma"}
    iv = bs.implied_volatility(target, **args, option_type=option_type)
    assert iv == pytest.approx(P["sigma"], abs=1e-8)


def test_vectorised_over_strikes():
    strikes = np.array([80.0, 100.0, 120.0])
    prices = bs.price(P["S"], strikes, P["T"], P["r"], P["sigma"])
    assert prices.shape == (3,)
    assert np.all(np.diff(prices) < 0)  # a call is worth less as K rises


def test_invalid_option_type():
    with pytest.raises(ValueError):
        bs.price(**P, option_type="straddle")