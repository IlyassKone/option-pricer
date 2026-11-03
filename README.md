# option-pricer

![CI](https://github.com/TON-PSEUDO/option-pricer/actions/workflows/ci.yml/badge.svg)

European and Asian option pricing library in Python, with a parallel C++ Monte Carlo engine exposed through pybind11.

Every model is checked against an independent reference: closed-form prices, put-call parity, finite-difference greeks, and statistical convergence tests.

## Features

| Module | Content |
| --- | --- |
| `black_scholes` | Black-Scholes-Merton price with continuous dividend yield, analytical greeks (delta, gamma, vega, theta, rho), implied volatility (Brent's method), vectorised with numpy |
| `monte_carlo` | Monte Carlo pricing with three estimators (plain, antithetic variates, control variate), GBM path simulation, arithmetic Asian options |
| `fast` | Same API backed by a C++17 engine: OpenMP multithreading, constant memory, results reproducible regardless of thread count |

## Results

### Variance reduction

ATM European call (S = K = 100, T = 1, r = 5 %, σ = 20 %), 100,000 paths. Black-Scholes reference: **10.4506**.

| Estimator | Price | Standard error | Variance reduction |
| --- | --- | --- | --- |
| Plain Monte Carlo | 10.358 | 0.046 | 1× |
| Antithetic variates | 10.401 | 0.033 | ≈ 2× |
| Control variate (discounted S_T) | 10.428 | 0.018 | ≈ 7× |

The control variate gives the accuracy of roughly 700,000 plain paths for the cost of 100,000.

### C++ engine vs numpy

16-core machine, best of 3 runs.

| Case | numpy | C++ (OpenMP) | Speed-up |
| --- | --- | --- | --- |
| European call, 2M paths | 22.9 ms | 3.1 ms | 7.4× |
| Asian call, 100k paths × 252 steps | 354.9 ms | 36.4 ms | 9.8× |

numpy is already vectorised C code, so single-threaded C++ is about as fast. The gains come from:

- **Parallelism**: paths are split into 64 chunks, each with its own random stream, and merged in a fixed order. The price is identical whatever the number of threads.
- **Memory**: numpy stores the full 100k × 252 path matrix (about 200 MB plus temporaries). The C++ engine generates one path at a time.
- **GIL release**: other Python threads keep running during a C++ pricing.

## Installation

Requires Linux or WSL2, a C++17 compiler, CMake and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/TON-PSEUDO/option-pricer.git
cd option-pricer
uv sync            # builds the C++ extension
uv run pytest -v   # 29 tests
```

## Usage

```python
from option_pricer import black_scholes as bs, monte_carlo as mc, fast

params = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)

bs.price(**params, option_type="call")           # 10.4506
bs.delta(**params, option_type="call")           # 0.6368
bs.implied_volatility(10.4506, S=100.0, K=100.0, T=1.0, r=0.05)  # 0.20

res = mc.price_european(**params, n_paths=100_000, method="control", seed=1)
res.price, res.std_error, res.conf_int_95

fast.price_asian_arithmetic(**params, n_steps=252, n_paths=100_000, seed=0)
```

Benchmark:

```bash
uv run python benchmarks/benchmark_mc.py
```

## Project structure

```
src/option_pricer/
    black_scholes.py     closed-form pricing, greeks, implied vol
    monte_carlo.py       numpy Monte Carlo, variance reduction, Asian options
    fast.py              Python API over the C++ engine
cpp/
    include/mc_pricer.hpp
    src/mc_pricer.cpp    parallel Monte Carlo engine (OpenMP)
    src/bindings.cpp     pybind11 module option_pricer._core
tests/                   pytest suite
benchmarks/              numpy vs C++ timing
```

## Validation

- Reference call price matches the textbook value (Hull) to 1e-4.
- Put-call parity holds with dividends: C − P = S·e^(−qT) − K·e^(−rT).
- Each analytical greek matches a central finite-difference derivative.
- Implied volatility recovers the input σ to 1e-8.
- Monte Carlo prices fall within 4 standard errors of Black-Scholes, for every estimator and for both engines.
- Simulated paths satisfy the risk-neutral martingale condition E[S_T] = S·e^(rT).

## Limitations

- Constant volatility and rates: no smile, no term structure.
- European and arithmetic Asian payoffs only, no early exercise.
- The C++ engine uses `std::mt19937_64` and `std::normal_distribution`, chosen for clarity rather than raw speed.

## Roadmap

- [ ] Binomial tree and finite-difference PDE pricers, American options
- [ ] Geometric Asian control variate for the arithmetic Asian
- [ ] Quasi-random (Sobol) sequences
- [ ] Heston model
