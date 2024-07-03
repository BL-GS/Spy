target("spy_perf")
    set_kind("headeronly")

    set_languages("c17", "c++20")
    set_warnings("all")
    
    add_includedirs("include", {public = true})

    add_deps("spy_util")
    add_packages("fmt", "spdlog", "magic_enum")
target_end()