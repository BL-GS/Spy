
target("spy_gpu")
    set_kind("static")
    set_default(false)
    set_toolchains("cuda")

    -- generate SASS code for SM architecture of current host
    add_cugencodes("native")
    -- generate PTX code for the virtual architecture to guarantee compatibility
    add_cugencodes("compute_75")    

    add_includedirs("include")
    add_files("backend/*.cu") 

    add_deps("spy_interface")
    add_packages("fmt", "magic-enum", "argparse")
target_end()
