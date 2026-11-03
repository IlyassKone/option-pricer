#include <pybind11/pybind11.h>

#include <string>

#include "mc_pricer.hpp"

namespace py = pybind11;
using namespace option_pricer;

namespace {

OptionType parse_type(const std::string& s) {
    if (s == "call") return OptionType::Call;
    if (s == "put") return OptionType::Put;
    throw py::value_error("option_type must be 'call' or 'put'");
}

py::dict to_dict(const MCResult& r) {
    py::dict d;
    d["price"] = r.price;
    d["std_error"] = r.std_error;
    d["n_paths"] = r.n_paths;
    return d;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ Monte Carlo engine for option pricing";

    m.def(
        "price_european",
        [](double S, double K, double T, double r, double sigma, double q,
           const std::string& option_type, std::size_t n_paths, std::uint64_t seed,
           bool antithetic) {
            const OptionType type = parse_type(option_type);  // needs the GIL
            MCResult res;
            {
                py::gil_scoped_release release;  // let other Python threads run
                res = price_european({S, r, sigma, q}, K, T, type, n_paths, seed,
                                     antithetic);
            }
            return to_dict(res);
        },
        py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"),
        py::arg("q") = 0.0, py::arg("option_type") = "call",
        py::arg("n_paths") = 100000, py::arg("seed") = 0, py::arg("antithetic") = false);

    m.def(
        "price_asian_arithmetic",
        [](double S, double K, double T, double r, double sigma, double q,
           const std::string& option_type, std::size_t n_steps, std::size_t n_paths,
           std::uint64_t seed) {
            const OptionType type = parse_type(option_type);
            MCResult res;
            {
                py::gil_scoped_release release;
                res = price_asian_arithmetic({S, r, sigma, q}, K, T, type, n_steps,
                                             n_paths, seed);
            }
            return to_dict(res);
        },
        py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"),
        py::arg("q") = 0.0, py::arg("option_type") = "call",
        py::arg("n_steps") = 252, py::arg("n_paths") = 50000, py::arg("seed") = 0);
}
