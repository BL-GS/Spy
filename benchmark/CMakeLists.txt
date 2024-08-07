cmake_minimum_required(VERSION 3.20)
project(SpyBenchmark)

find_package(benchmark CONFIG REQUIRED)

# Project setup
macro(add_spy_benchmark BENCHMARK_NAME)
    add_executable(benchmark_${BENCHMARK_NAME} ${PROJECT_SOURCE_DIR}/src/${BENCHMARK_NAME}.cpp)
    target_include_directories(benchmark_${BENCHMARK_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(benchmark_${BENCHMARK_NAME} PRIVATE
        # Internal library
        SpyUtil
        SpyPerf
        SpyOrchestration
        SpyStorage
        SpyModel
        SpyBackend
        # External libraries
        benchmark::benchmark
        benchmark::benchmark_main
    )
endmacro()

add_spy_benchmark(temp)