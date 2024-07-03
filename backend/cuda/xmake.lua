target("spy_backend_cuda")
    set_kind("static")
    set_default(false)
    add_toolchains("cuda")

    set_languages("c17", "c++20")
    set_warnings("all")

    add_values("cuda.build.devlink", true)
    -- generate SASS code for SM architecture of current host
    add_cugencodes("native")
    -- generate PTX code for the virtual architecture to guarantee compatibility
    add_cugencodes("compute_75")

    add_includedirs("include")
    add_files("operator/**.cu")

    add_deps("spy_util", "spy_backend_interface")
    add_packages("fmt", "spdlog", "magic_enum")
    add_links("cublas", "cudart", "cublasLt", "culibos", "cuda")
    add_linkdirs("/usr/local/cuda/lib64/stubs")
    -- add_syslinks()
target_end()
