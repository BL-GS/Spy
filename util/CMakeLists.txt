CMAKE_MINIMUM_REQUIRED(VERSION 3.20)
PROJECT(SpyUtil)

# Library
FIND_PACKAGE(fmt	 	CONFIG REQUIRED)
FIND_PACKAGE(spdlog		CONFIG REQUIRED)
FIND_PACKAGE(magic_enum CONFIG REQUIRED)
FIND_PACKAGE(argparse 	CONFIG REQUIRED)

FILE(GLOB_RECURSE source_files ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Project setup
ADD_LIBRARY(${PROJECT_NAME} INTERFACE ${source_files})
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} INTERFACE ${PROJECT_SOURCE_DIR}/include)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} INTERFACE
    # External libraries
    fmt::fmt-header-only
    spdlog::spdlog_header_only
    argparse::argparse
    magic_enum::magic_enum
)