"""Speed comparison: numpy (vectorised) vs C++ (pybind11 + OpenMP).

Run with:  uv run python benchmarks/benchmark_mc.py
"""

import os
import time

from option_pricer import fast
from option_pricer import monte_carlo as mc

P = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)


def best_time(fn, repeats=3):
    """Best wall-clock time over several runs, in milliseconds."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best * 1000, result


CASES = [
    ("European call, 2M paths",
     lambda: mc.price_european(**P, n_paths=2_000_000, seed=1),
     lambda: fast.price_european(**P, n_paths=2_000_000, seed=1)),
    ("Asian call, 100k paths x 252 steps",
     lambda: mc.price_asian_arithmetic(**P, n_steps=252, n_paths=100_000, seed=0),
     lambda: fast.price_asian_arithmetic(**P, n_steps=252, n_paths=100_000, seed=0)),
]

if __name__ == "__main__":
    print(f"CPU cores available: {os.cpu_count()}\n")
    print(f"{'Case':<38}{'numpy (ms)':>12}{'C++ (ms)':>12}{'Speed-up':>10}")
    for name, numpy_fn, cpp_fn in CASES:
        t_np, r_np = best_time(numpy_fn)
        t_cpp, r_cpp = best_time(cpp_fn)
        print(f"{name:<38}{t_np:>12.1f}{t_cpp:>12.1f}{t_np / t_cpp:>9.1f}x")
        print(f"{'':<38}price {r_np.price:.4f} vs {r_cpp.price:.4f}")
