project(SpyTest)

find_package(GTest 		CONFIG REQUIRED)
find_package(fmt	 	CONFIG REQUIRED)
find_package(magic_enum CONFIG REQUIRED)
find_package(argparse 	CONFIG REQUIRED)

enable_testing()

macro(add_spy_test target_name sources_file)
    add_executable(${target_name} ${sources_file})
    target_include_directories(${target_name} PRIVATE ${include_dir})
    target_link_libraries(${target_name} PRIVATE 
        fmt::fmt-header-only
        argparse::argparse
        magic_enum::magic_enum
        GTest::gtest
        GTest::gtest_main
    )
    add_test(NAME test_${target_name} COMMAND ${target_name})
endmacro(add_spy_test)

add_spy_test(operator_basic "operator/basic.cpp")
add_spy_test(operator_mat "operator/mat.cpp")
