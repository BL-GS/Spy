CMAKE_MINIMUM_REQUIRED(VERSION 3.20)

PROJECT(spy_cpu)

IF (NOT ${spy_interface})
    MESSAGE(FATAL_ERROR "Variable spy_interface hasn't been set")
ENDIF ()

FIND_PACKAGE(fmt	 	CONFIG REQUIRED)
FIND_PACKAGE(magic_enum CONFIG REQUIRED)
FIND_PACKAGE(argparse 	CONFIG REQUIRED)
FIND_PACKAGE(unofficial-concurrentqueue CONFIG REQUIRED)

FILE(GLOB_RECURSE backend_files ${PROJECT_SOURCE_DIR}/backend/*.cpp)
FILE(GLOB_RECURSE operator_files ${PROJECT_SOURCE_DIR}/operator/*.cpp)

ADD_LIBRARY(${PROJECT_NAME} ${backend_files} ${operator_files})
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PRIVATE ${spy_include_dir})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE
    # External library
    fmt::fmt-header-only
    argparse::argparse
    magic_enum::magic_enum
    unofficial::concurrentqueue::concurrentqueue
    # Internal library
    ${spy_interface}
)