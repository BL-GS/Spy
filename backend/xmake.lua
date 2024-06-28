option("enable-cuda")
    set_default(false)
    set_showmenu(true)
option_end()

target("spy_backend")
    set_kind("static")
    set_default(false)

    set_languages("c17", "c++20")
    set_warnings("all")

    add_deps("spy_backend_interface", "spy_backend_host")

    if has_config("enable-cuda") then
        add_defines("SPY_BACKEND_CUDA")
        add_deps("spy_backend_cuda")
    end

    add_packages("magic_enum", "concurrentqueue")
target_end()