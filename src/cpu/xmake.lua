
target("spy_cpu")
    set_kind("static")
    set_default(false)

    add_includedirs("include")
    add_files("operator/*.cpp", "backend/*.cpp")

    add_deps("spy_interface")
    add_packages("fmt", "spdlog", "magic_enum", "argparse", "concurrentqueue")
target_end()