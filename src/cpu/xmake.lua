
target("spy_cpu")
    set_kind("static")
    set_default(false)

    add_files("operator/*.cpp", "backend/*.cpp")

    add_deps("spy_interface")
    add_packages("fmt", "magic_enum", "argparse", "concurrentqueue")
target_end()