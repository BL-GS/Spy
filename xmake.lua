set_project("spy")
set_version("0.1.0")

-- Build Type
add_rules("mode.release", "mode.debug")

-- Vector Extension
add_cxxflags("-mf16c")
add_vectorexts("avx", "avx2")
add_vectorexts("sse", "sse2", "sse3", "ssse3", "sse4.2")
add_vectorexts("fma")
add_vectorexts("all")

-- Language
set_languages("c17", "c++20")
set_warnings("all")

-- External library 
add_requires("liburing")
add_requires("fmt", "spdlog", "argparse", "magic_enum", "concurrentqueue")
add_requires("gtest")

-- Internal library

-- Submodules
includes("util")
includes("perf")
includes("orchestration")
includes("backend")

-- Main Project
target("spy")
    set_kind("binary")
    set_default(true)

    add_includedirs("include")
    add_files("spy.cpp")

    add_deps("spy_util", "spy_perf", "spy_orchestration", "spy_backend")

    add_packages("liburing")
    add_packages("fmt", "spdlog", "magic_enum", "concurrentqueue")
target_end()
