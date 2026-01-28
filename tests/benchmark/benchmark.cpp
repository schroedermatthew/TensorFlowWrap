// benchmark.cpp
// Performance benchmarks for TensorFlowWrap
//
// Build:
//   g++ -std=c++20 -O3 -DNDEBUG \
//       -I include -I third_party/tf_stub \
//       -DTF_WRAPPER_TF_STUB_ENABLED=1 \
//       third_party/tf_stub/tf_c_stub.cpp \
//       tests/benchmark/benchmark.cpp \
//       -o benchmark
//
// Build with real TF:
//   g++ -std=c++20 -O3 -DNDEBUG \
//       -I include -I tensorflow_c/include \
//       tests/benchmark/benchmark.cpp \
//       -L tensorflow_c/lib -ltensorflow \
//       -Wl,-rpath,tensorflow_c/lib \
//       -pthread -o benchmark
//
// Run:
//   ./benchmark [--iterations N] [--warmup N] [--json]

#include "tf_wrap/tensor.hpp"
#include "tf_wrap/session.hpp"
#include "tf_wrap/graph.hpp"
#include "tf_wrap/small_vector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace tf_wrap;
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::nano>;

// ============================================================================
// Benchmark Infrastructure
// ============================================================================

struct BenchmarkResult {
    std::string name;
    double min_ns;
    double max_ns;
    double mean_ns;
    double median_ns;
    double stddev_ns;
    std::size_t iterations;
    std::size_t ops_per_iter;
};

class Benchmark {
public:
    Benchmark(const std::string& name, std::size_t ops_per_iter = 1)
        : name_(name), ops_per_iter_(ops_per_iter) {}

    template<typename Func>
    BenchmarkResult run(Func&& func, std::size_t iterations, std::size_t warmup) {
        // Warmup
        for (std::size_t i = 0; i < warmup; ++i) {
            func();
        }

        // Benchmark
        std::vector<double> times;
        times.reserve(iterations);

        for (std::size_t i = 0; i < iterations; ++i) {
            auto start = Clock::now();
            func();
            auto end = Clock::now();
            times.push_back(std::chrono::duration_cast<Duration>(end - start).count());
        }

        // Calculate statistics
        std::sort(times.begin(), times.end());

        double min_ns = times.front();
        double max_ns = times.back();
        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        double mean_ns = sum / times.size();
        double median_ns = times[times.size() / 2];

        double sq_sum = 0.0;
        for (double t : times) {
            sq_sum += (t - mean_ns) * (t - mean_ns);
        }
        double stddev_ns = std::sqrt(sq_sum / times.size());

        return BenchmarkResult{
            name_, min_ns, max_ns, mean_ns, median_ns, stddev_ns,
            iterations, ops_per_iter_
        };
    }

private:
    std::string name_;
    std::size_t ops_per_iter_;
};

// ============================================================================
// Result Reporting
// ============================================================================

void print_result(const BenchmarkResult& r, bool json = false) {
    if (json) {
        std::cout << "  {\"name\": \"" << r.name << "\", "
                  << "\"min_ns\": " << std::fixed << std::setprecision(2) << r.min_ns << ", "
                  << "\"mean_ns\": " << r.mean_ns << ", "
                  << "\"median_ns\": " << r.median_ns << ", "
                  << "\"max_ns\": " << r.max_ns << ", "
                  << "\"stddev_ns\": " << r.stddev_ns << ", "
                  << "\"iterations\": " << r.iterations << ", "
                  << "\"ops_per_iter\": " << r.ops_per_iter << "}";
    } else {
        double ops_per_sec = (r.ops_per_iter * 1e9) / r.mean_ns;
        std::cout << std::left << std::setw(40) << r.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(1) << r.mean_ns << " ns"
                  << std::setw(12) << r.median_ns << " ns"
                  << std::setw(12) << r.stddev_ns << " ns"
                  << std::setw(14) << std::setprecision(0) << ops_per_sec << " ops/s"
                  << "\n";
    }
}

void print_header() {
    std::cout << std::left << std::setw(40) << "Benchmark"
              << std::right << std::setw(12) << "Mean"
              << std::setw(12) << "Median"
              << std::setw(12) << "StdDev"
              << std::setw(14) << "Throughput"
              << "\n";
    std::cout << std::string(90, '-') << "\n";
}

// ============================================================================
// Tensor Benchmarks
// ============================================================================

std::vector<BenchmarkResult> run_tensor_benchmarks(std::size_t iterations, std::size_t warmup) {
    std::vector<BenchmarkResult> results;

    // FromScalar<float>
    {
        Benchmark b("Tensor::FromScalar<float>");
        auto r = b.run([&]() {
            auto t = Tensor::FromScalar<float>(3.14f);
            (void)t;
        }, iterations, warmup);
        results.push_back(r);
    }

    // FromScalar<double>
    {
        Benchmark b("Tensor::FromScalar<double>");
        auto r = b.run([&]() {
            auto t = Tensor::FromScalar<double>(3.14159265359);
            (void)t;
        }, iterations, warmup);
        results.push_back(r);
    }

    // FromVector small (10 elements)
    {
        std::vector<float> data(10, 1.0f);
        Benchmark b("Tensor::FromVector<float>[10]");
        auto r = b.run([&]() {
            auto t = Tensor::FromVector<float>({10}, data);
            (void)t;
        }, iterations, warmup);
        results.push_back(r);
    }

    // FromVector medium (1000 elements)
    {
        std::vector<float> data(1000, 1.0f);
        Benchmark b("Tensor::FromVector<float>[1000]");
        auto r = b.run([&]() {
            auto t = Tensor::FromVector<float>({1000}, data);
            (void)t;
        }, iterations, warmup);
        results.push_back(r);
    }

    // FromVector large (100000 elements)
    {
        std::vector<float> data(100000, 1.0f);
        Benchmark b("Tensor::FromVector<float>[100000]");
        auto r = b.run([&]() {
            auto t = Tensor::FromVector<float>({100000}, data);
            (void)t;
        }, iterations, warmup);
        results.push_back(r);
    }

    // Zeros
    {
        Benchmark b("Tensor::Zeros<float>[1000]");
        auto r = b.run([&]() {
            auto t = Tensor::Zeros<float>({1000});
            (void)t;
        }, iterations, warmup);
        results.push_back(r);
    }

    // Clone
    {
        auto src = Tensor::FromVector<float>({1000}, std::vector<float>(1000, 1.0f));
        Benchmark b("Tensor::Clone[1000]");
        auto r = b.run([&]() {
            auto t = src.Clone();
            (void)t;
        }, iterations, warmup);
        results.push_back(r);
    }

    // Read view
    {
        auto t = Tensor::FromVector<float>({1000}, std::vector<float>(1000, 1.0f));
        Benchmark b("Tensor::read<float>[1000]");
        auto r = b.run([&]() {
            auto view = t.read<float>();
            (void)view[0];
        }, iterations, warmup);
        results.push_back(r);
    }

    // ToScalar
    {
        auto t = Tensor::FromScalar<float>(3.14f);
        Benchmark b("Tensor::ToScalar<float>");
        auto r = b.run([&]() {
            float v = t.ToScalar<float>();
            (void)v;
        }, iterations, warmup);
        results.push_back(r);
    }

    // ToVector
    {
        auto t = Tensor::FromVector<float>({1000}, std::vector<float>(1000, 1.0f));
        Benchmark b("Tensor::ToVector<float>[1000]");
        auto r = b.run([&]() {
            auto v = t.ToVector<float>();
            (void)v;
        }, iterations, warmup);
        results.push_back(r);
    }

    // Move construction
    {
        Benchmark b("Tensor move construction");
        auto r = b.run([&]() {
            auto t1 = Tensor::FromScalar<float>(1.0f);
            auto t2 = std::move(t1);
            (void)t2;
        }, iterations, warmup);
        results.push_back(r);
    }

    // FromString
    {
        Benchmark b("Tensor::FromString");
        auto r = b.run([&]() {
            auto t = Tensor::FromString("hello world");
            (void)t;
        }, iterations, warmup);
        results.push_back(r);
    }

    return results;
}

// ============================================================================
// SmallVector Benchmarks
// ============================================================================

std::vector<BenchmarkResult> run_smallvector_benchmarks(std::size_t iterations, std::size_t warmup) {
    std::vector<BenchmarkResult> results;

    // push_back (inline)
    {
        Benchmark b("SmallVector<int,8> push_back x4", 4);
        auto r = b.run([&]() {
            SmallVector<int, 8> v;
            v.push_back(1);
            v.push_back(2);
            v.push_back(3);
            v.push_back(4);
        }, iterations, warmup);
        results.push_back(r);
    }

    // push_back (spill to heap)
    {
        Benchmark b("SmallVector<int,4> push_back x8 (spill)", 8);
        auto r = b.run([&]() {
            SmallVector<int, 4> v;
            for (int i = 0; i < 8; ++i) {
                v.push_back(i);
            }
        }, iterations, warmup);
        results.push_back(r);
    }

    // std::vector comparison
    {
        Benchmark b("std::vector<int> push_back x4", 4);
        auto r = b.run([&]() {
            std::vector<int> v;
            v.push_back(1);
            v.push_back(2);
            v.push_back(3);
            v.push_back(4);
        }, iterations, warmup);
        results.push_back(r);
    }

    // reserve + push_back
    {
        Benchmark b("SmallVector<int,8> reserve(100) + push x100", 100);
        auto r = b.run([&]() {
            SmallVector<int, 8> v;
            v.reserve(100);
            for (int i = 0; i < 100; ++i) {
                v.push_back(i);
            }
        }, iterations, warmup);
        results.push_back(r);
    }

    // Copy
    {
        SmallVector<int, 8> src;
        for (int i = 0; i < 8; ++i) src.push_back(i);
        Benchmark b("SmallVector<int,8> copy (inline)");
        auto r = b.run([&]() {
            SmallVector<int, 8> dst = src;
            (void)dst;
        }, iterations, warmup);
        results.push_back(r);
    }

    // Move
    {
        Benchmark b("SmallVector<int,8> move");
        auto r = b.run([&]() {
            SmallVector<int, 8> src;
            for (int i = 0; i < 4; ++i) src.push_back(i);
            SmallVector<int, 8> dst = std::move(src);
            (void)dst;
        }, iterations, warmup);
        results.push_back(r);
    }

    return results;
}

// ============================================================================
// Session/Graph Benchmarks
// ============================================================================

std::vector<BenchmarkResult> run_session_benchmarks(std::size_t iterations, std::size_t warmup) {
    std::vector<BenchmarkResult> results;

    // Graph construction
    {
        Benchmark b("Graph construction");
        auto r = b.run([&]() {
            Graph g;
            (void)g;
        }, iterations, warmup);
        results.push_back(r);
    }

    // SessionOptions construction
    {
        Benchmark b("SessionOptions construction");
        auto r = b.run([&]() {
            SessionOptions opts;
            (void)opts;
        }, iterations, warmup);
        results.push_back(r);
    }

    // Session construction
    {
        Graph g;
        Benchmark b("Session construction");
        auto r = b.run([&]() {
            SessionOptions opts;
            Session s(g, opts);
            (void)s;
        }, iterations, warmup);
        results.push_back(r);
    }

    // Status construction
    {
        Benchmark b("Status construction");
        auto r = b.run([&]() {
            Status st;
            (void)st;
        }, iterations, warmup);
        results.push_back(r);
    }

    // Buffer construction
    {
        Benchmark b("Buffer construction");
        auto r = b.run([&]() {
            Buffer buf;
            (void)buf;
        }, iterations, warmup);
        results.push_back(r);
    }

    return results;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::size_t iterations = 10000;
    std::size_t warmup = 1000;
    bool json_output = false;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = std::stoul(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = std::stoul(argv[++i]);
        } else if (std::strcmp(argv[i], "--json") == 0) {
            json_output = true;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --iterations N  Number of iterations (default: 10000)\n"
                      << "  --warmup N      Warmup iterations (default: 1000)\n"
                      << "  --json          Output results as JSON\n";
            return 0;
        }
    }

    std::vector<BenchmarkResult> all_results;

    if (!json_output) {
        std::cout << "=== TensorFlowWrap Benchmarks ===\n";
        std::cout << "Iterations: " << iterations << ", Warmup: " << warmup << "\n\n";
    }

    // Run benchmarks
    auto tensor_results = run_tensor_benchmarks(iterations, warmup);
    auto smallvec_results = run_smallvector_benchmarks(iterations, warmup);
    auto session_results = run_session_benchmarks(iterations, warmup);

    all_results.insert(all_results.end(), tensor_results.begin(), tensor_results.end());
    all_results.insert(all_results.end(), smallvec_results.begin(), smallvec_results.end());
    all_results.insert(all_results.end(), session_results.begin(), session_results.end());

    if (json_output) {
        std::cout << "{\n";
        std::cout << "  \"iterations\": " << iterations << ",\n";
        std::cout << "  \"warmup\": " << warmup << ",\n";
        std::cout << "  \"results\": [\n";
        for (std::size_t i = 0; i < all_results.size(); ++i) {
            print_result(all_results[i], true);
            if (i + 1 < all_results.size()) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "  ]\n";
        std::cout << "}\n";
    } else {
        std::cout << "--- Tensor Operations ---\n";
        print_header();
        for (const auto& r : tensor_results) {
            print_result(r);
        }

        std::cout << "\n--- SmallVector Operations ---\n";
        print_header();
        for (const auto& r : smallvec_results) {
            print_result(r);
        }

        std::cout << "\n--- Session/Graph Operations ---\n";
        print_header();
        for (const auto& r : session_results) {
            print_result(r);
        }

        std::cout << "\n=== Benchmark Complete ===\n";
    }

    return 0;
}
