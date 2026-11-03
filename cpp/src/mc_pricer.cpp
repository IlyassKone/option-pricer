#include "mc_pricer.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace option_pricer {
namespace {

// Paths are split into a fixed number of chunks, each with its own random
// stream. Threads pick up chunks, and the chunk results are merged in a fixed
// order: the price is identical whatever the number of threads.
constexpr std::size_t kChunks = 64;

double payoff(double s, double strike, OptionType type) {
    return type == OptionType::Call ? std::max(s - strike, 0.0)
                                    : std::max(strike - s, 0.0);
}

// Running sum and sum of squares, mergeable across chunks.
struct Accumulator {
    double sum = 0.0;
    double sum_sq = 0.0;
    std::size_t n = 0;

    void add(double x) {
        sum += x;
        sum_sq += x * x;
        ++n;
    }
    void merge(const Accumulator& other) {
        sum += other.sum;
        sum_sq += other.sum_sq;
        n += other.n;
    }
    MCResult result(double discount) const {
        const double count = static_cast<double>(n);
        const double mean = sum / count;
        const double var = (sum_sq - count * mean * mean) / (count - 1.0);
        return {discount * mean, discount * std::sqrt(var / count), n};
    }
};

std::mt19937_64 make_rng(std::uint64_t seed, std::size_t chunk) {
    std::seed_seq seq{static_cast<std::uint32_t>(seed),
                      static_cast<std::uint32_t>(seed >> 32),
                      static_cast<std::uint32_t>(chunk)};
    return std::mt19937_64(seq);
}

// Number of samples handled by chunk c when n samples are split evenly.
std::size_t chunk_size(std::size_t n, std::size_t c) {
    return n / kChunks + (c < n % kChunks ? 1 : 0);
}

void validate(const MarketParams& mkt, double strike, double maturity,
              std::size_t n_paths) {
    if (mkt.spot <= 0 || strike <= 0 || maturity <= 0 || mkt.vol <= 0)
        throw std::invalid_argument("spot, strike, maturity and vol must be > 0");
    if (n_paths < 2) throw std::invalid_argument("n_paths must be >= 2");
}

MCResult merge_all(const std::vector<Accumulator>& parts, double discount) {
    Accumulator total;
    for (const auto& p : parts) total.merge(p);
    return total.result(discount);
}

}  // namespace

MCResult price_european(const MarketParams& mkt, double strike, double maturity,
                        OptionType type, std::size_t n_paths, std::uint64_t seed,
                        bool antithetic) {
    validate(mkt, strike, maturity, n_paths);
    const double drift = (mkt.rate - mkt.div_yield - 0.5 * mkt.vol * mkt.vol) * maturity;
    const double diffusion = mkt.vol * std::sqrt(maturity);
    const std::size_t n_samples = antithetic ? n_paths / 2 : n_paths;

    std::vector<Accumulator> parts(kChunks);
#pragma omp parallel for schedule(dynamic)
    for (std::size_t c = 0; c < kChunks; ++c) {
        auto rng = make_rng(seed, c);
        std::normal_distribution<double> normal(0.0, 1.0);
        Accumulator acc;
        for (std::size_t i = 0; i < chunk_size(n_samples, c); ++i) {
            const double z = normal(rng);
            const double up = mkt.spot * std::exp(drift + diffusion * z);
            if (antithetic) {
                const double down = mkt.spot * std::exp(drift - diffusion * z);
                acc.add(0.5 * (payoff(up, strike, type) + payoff(down, strike, type)));
            } else {
                acc.add(payoff(up, strike, type));
            }
        }
        parts[c] = acc;
    }
    return merge_all(parts, std::exp(-mkt.rate * maturity));
}

MCResult price_asian_arithmetic(const MarketParams& mkt, double strike, double maturity,
                                OptionType type, std::size_t n_steps,
                                std::size_t n_paths, std::uint64_t seed) {
    validate(mkt, strike, maturity, n_paths);
    if (n_steps == 0) throw std::invalid_argument("n_steps must be >= 1");
    const double dt = maturity / static_cast<double>(n_steps);
    const double drift = (mkt.rate - mkt.div_yield - 0.5 * mkt.vol * mkt.vol) * dt;
    const double diffusion = mkt.vol * std::sqrt(dt);
    const double log_spot = std::log(mkt.spot);

    std::vector<Accumulator> parts(kChunks);
#pragma omp parallel for schedule(dynamic)
    for (std::size_t c = 0; c < kChunks; ++c) {
        auto rng = make_rng(seed, c);
        std::normal_distribution<double> normal(0.0, 1.0);
        Accumulator acc;
        for (std::size_t i = 0; i < chunk_size(n_paths, c); ++i) {
            double log_s = log_spot;
            double running_sum = 0.0;
            for (std::size_t j = 0; j < n_steps; ++j) {
                log_s += drift + diffusion * normal(rng);
                running_sum += std::exp(log_s);
            }
            acc.add(payoff(running_sum / static_cast<double>(n_steps), strike, type));
        }
        parts[c] = acc;
    }
    return merge_all(parts, std::exp(-mkt.rate * maturity));
}

}  // namespace option_pricer
