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


-- Library 
add_requires("liburing")
add_requires("fmt", "spdlog", "argparse", "magic_enum", "concurrentqueue")
add_requires("gtest")

-- Interface
target("spy_interface")
    set_kind("headeronly")
    add_headerfiles("include/**.h")
    add_includedirs("include", {public = true})
target_end()

-- Backends
includes("src/cpu")
includes("src/gpu")

option("enable-cpu")
    set_default(true)
    set_showmenu(true)
option_end()
option("enable-gpu")
    set_default(false)
    set_showmenu(true)
option_end()

-- Main Project
target("spy")
    set_kind("binary")
    set_default(true)

    add_includedirs("include")
    add_files("spy.cpp")

    if has_config("enable-cpu") then 
        add_deps("spy_cpu")
    end
    if has_config("enable-gpu") then
        add_deps("spy_gpu")
    end    

    add_packages("liburing")
    add_packages("fmt", "spdlog", "magic_enum", "argparse", "concurrentqueue")
target_end()

