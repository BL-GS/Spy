add_rules("mode.release", "mode.debug")
set_languages("c17", "c++20")

add_requires("fmt", "argparse", "gtest", "magic-enum")

target("spy")
    set_kind("binary")
    add_cxxflags("-march=native", "-mf16c", "-msse2", "-msse3", "-mavx", "-mavx2", "-mavx512f", "-Wall")
    add_includedirs("include")
    add_files("spy.cpp")
    add_files("src/**.cpp") 
    add_packages("fmt", "magic-enum", "argparse")