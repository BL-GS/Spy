target("spy_util")
    set_kind("headeronly")
    set_default(false)

    set_languages("c17", "c++20")
    set_warnings("all")
    
    add_includedirs("include", {public = true})

    add_packages("fmt", "spdlog", "magic_enum", "argparse")
target_end()