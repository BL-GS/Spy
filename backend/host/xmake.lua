target("spy_backend_host")
    set_kind("static")
    set_default(false)

    set_languages("c17", "c++20")
    set_warnings("all")
    
    add_includedirs("include")
    add_files("operator/*.cpp")

    add_deps("spy_util", "spy_backend_interface")
    add_packages("fmt", "spdlog", "magic_enum", "concurrentqueue")
target_end()