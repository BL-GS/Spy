cmake_minimum_required(VERSION 3.20)

project(spy_cpu)

IF (NOT ${spy_interface})
    MESSAGE(FATAL_ERROR "Variable spy_interface hasn't been set")
ENDIF ()

find_package(fmt	 	CONFIG REQUIRED)
find_package(magic_enum CONFIG REQUIRED)
find_package(argparse 	CONFIG REQUIRED)
find_package(unofficial-concurrentqueue CONFIG REQUIRED)

FILE(GLOB_RECURSE backend_files ${PROJECT_SOURCE_DIR}/backend/*.cpp)
FILE(GLOB_RECURSE operator_files ${PROJECT_SOURCE_DIR}/operator/*.cpp)

add_library(${PROJECT_NAME} ${backend_files} ${operator_files})
target_include_directories(${PROJECT_NAME} PRIVATE ${spy_include_dir})
target_link_libraries(${PROJECT_NAME} PRIVATE
    # External library
    fmt::fmt-header-only
    argparse::argparse
    magic_enum::magic_enum
    unofficial::concurrentqueue::concurrentqueue
    # Internal library
    ${spy_interface}
)