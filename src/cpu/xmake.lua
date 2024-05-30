target("spy_cpu")
    set_kind("static")
    set_default(false)

    set_languages("c17", "c++20")
    set_warnings("all")
    
    add_includedirs("include")
    add_files("operator/*.cpp", "backend/*.cpp")

    add_deps("spy_interface")
    add_packages("fmt", "spdlog", "magic_enum", "argparse", "concurrentqueue")
target_end()