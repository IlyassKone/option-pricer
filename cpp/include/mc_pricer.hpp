#pragma once

#include <cstddef>
#include <cstdint>

namespace option_pricer {

enum class OptionType { Call, Put };

struct MCResult {
    double price;
    double std_error;
    std::size_t n_paths;
};

struct MarketParams {
    double spot;
    double rate;
    double vol;
    double div_yield = 0.0;
};

// European option, exact simulation of S_T, optional antithetic variates.
MCResult price_european(const MarketParams& mkt, double strike, double maturity,
                        OptionType type, std::size_t n_paths, std::uint64_t seed,
                        bool antithetic = false);

// Arithmetic-average Asian option. Paths are generated one at a time:
// memory use is O(1), whatever the number of paths and steps.
MCResult price_asian_arithmetic(const MarketParams& mkt, double strike, double maturity,
                                OptionType type, std::size_t n_steps,
                                std::size_t n_paths, std::uint64_t seed);

}  // namespace option_pricer
