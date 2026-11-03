"""Python API over the C++ Monte Carlo engine (option_pricer._core).

Same signatures and MCResult return type as option_pricer.monte_carlo,
so the two engines can be swapped and compared directly.
"""

from typing import Literal

from option_pricer import _core
from option_pricer.black_scholes import OptionType
from option_pricer.monte_carlo import MCResult

FastMethod = Literal["plain", "antithetic"]


def _to_result(d: dict) -> MCResult:
    return MCResult(price=d["price"], std_error=d["std_error"], n_paths=d["n_paths"])


def price_european(S, K, T, r, sigma, q=0.0, option_type: OptionType = "call",
                   n_paths: int = 100_000, method: FastMethod = "plain",
                   seed: int = 0) -> MCResult:
    if method not in ("plain", "antithetic"):
        raise ValueError(f"unknown method {method!r}")
    return _to_result(_core.price_european(
        S, K, T, r, sigma, q, option_type, n_paths, seed, method == "antithetic"))


def price_asian_arithmetic(S, K, T, r, sigma, q=0.0,
                           option_type: OptionType = "call",
                           n_steps: int = 252, n_paths: int = 50_000,
                           seed: int = 0) -> MCResult:
    return _to_result(_core.price_asian_arithmetic(
        S, K, T, r, sigma, q, option_type, n_steps, n_paths, seed))
