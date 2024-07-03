includes("interface")
includes("host")

option("enable-cuda")
    set_default(false)
    set_showmenu(true)
    
    includes("cuda")
option_end()

target("spy_backend")
    set_kind("static")

    set_languages("c17", "c++20")
    set_warnings("all")

    add_deps("spy_util", "spy_backend_host")

    if has_config("enable-cuda") then
        add_defines("SPY_BACKEND_CUDA")
        add_deps("spy_backend_cuda")
    end

    add_packages("fmt", "spdlog", "magic_enum", "concurrentqueue")
target_end()