target("spy_util")
    set_kind("static")

    set_languages("c17", "c++20")
    set_warnings("all")
    
    add_includedirs("include", {public = true})
    add_files("src/**.cpp")

    add_packages("fmt", "spdlog", "magic_enum", "argparse")
target_end()