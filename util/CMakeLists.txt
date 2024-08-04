cmake_minimum_required(VERSION 3.20)
project(SpyUtil)

# Library
find_package(fmt	 	CONFIG REQUIRED)
find_package(spdlog		CONFIG REQUIRED)
find_package(magic_enum CONFIG REQUIRED)
find_package(argparse 	CONFIG REQUIRED)

file(GLOB_RECURSE source_files ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Project setup
add_library(${PROJECT_NAME} INTERFACE ${source_files})
target_include_directories(${PROJECT_NAME} INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} INTERFACE
    # External libraries
    fmt::fmt-header-only
    spdlog::spdlog_header_only
    argparse::argparse
    magic_enum::magic_enum
)