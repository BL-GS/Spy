#include <thread>
#include <benchmark/benchmark.h>

static void bench_pow(benchmark::State &state) {
    for (auto _: state) {
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(1s);
    }
}
BENCHMARK(bench_pow);

BENCHMARK_MAIN();