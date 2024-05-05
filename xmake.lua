add_rules("mode.release", "mode.debug")

add_requires("uring", {system = true})
add_requires("fmt", "argparse", "gtest", "magic-enum", "concurrentqueue")

target("spy_cpu")
    set_kind("static")
    set_languages("c17", "c++20")

    add_cxxflags("-mf16c")
    add_vectorexts("avx", "avx2")
    add_vectorexts("sse", "sse2", "sse3", "ssse3", "sse4.2")
    add_vectorexts("fma")
    set_warnings("all")

    add_includedirs("include")
    add_files("src/cpu/**.cpp") 

    add_packages("uring")
    add_packages("fmt", "magic-enum", "argparse", "concurrentqueue")
target_end()

option("enable-cpu")
    set_default(true)
    set_showmenu(true)
option_end()

target("spy_gpu")
    set_kind("static")
    set_toolchains("cuda")
    set_languages("c17", "c++20")

    add_cxxflags("-mf16c")
    add_vectorexts("avx", "avx2")
    add_vectorexts("sse", "sse2", "sse3", "ssse3", "sse4.2")
    add_vectorexts("fma")
    set_warnings("all")

    -- generate SASS code for SM architecture of current host
    add_cugencodes("native")
    -- generate PTX code for the virtual architecture to guarantee compatibility
    add_cugencodes("compute_75")    

    add_includedirs("include", "src/gpu/include")
    add_files("src/gpu/**.cu") 

    add_packages("fmt", "magic-enum", "argparse")
target_end()

option("enable-gpu")
    set_default(false)
    set_showmenu(true)
option_end()

target("spy")
    set_kind("binary")
    set_languages("c17", "c++20")

    add_cxxflags("-mf16c")
    add_vectorexts("avx", "avx2")
    add_vectorexts("sse", "sse2", "sse3", "ssse3", "sse4.2")
    add_vectorexts("fma")
    set_warnings("all")

    add_includedirs("include")
    add_files("spy.cpp")

    if has_config("enable-cpu") then 
        add_deps("spy_cpu")
    end
    if has_config("enable-gpu") then
        add_deps("spy_gpu")
    end    

    add_packages("uring")
    add_packages("fmt", "magic-enum", "argparse", "concurrentqueue")
target_end()

