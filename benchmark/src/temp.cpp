#include <thread>
#include <benchmark/benchmark.h>

static void bench_pow(benchmark::State &state) {
    for (auto _: state) {
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(1s);
    }
}
BENCHMARK(bench_pow);

int main(int argc, char **argv) {
    char arg0_default[] = "benchmark";
    char *args_default = arg0_default;
    if (!argv) {
        argc = 1;
        argv = &args_default;
    }
    benchmark ::Initialize(&argc, argv);
    if (benchmark ::ReportUnrecognizedArguments(argc, argv)) { return 1; }
    benchmark ::RunSpecifiedBenchmarks();
    benchmark ::Shutdown();
    return 0;
}