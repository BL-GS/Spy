cmake_minimum_required(VERSION 3.20)
project(spy)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)

include(CMakeArch.cmake)

if (CMAKE_BUILD_TYPE STREQUAL Release)
	ADD_DEFINITIONS(-DNDEBUG)
endif()

find_package(fmt	 	CONFIG REQUIRED)
find_package(magic_enum CONFIG REQUIRED)
find_package(argparse 	CONFIG REQUIRED)
find_package(unofficial-concurrentqueue CONFIG REQUIRED)

set(include_dir ${PROJECT_SOURCE_DIR}/include)
FILE(GLOB_RECURSE source_files ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_executable(spy spy.cpp ${source_files})
target_include_directories(spy PRIVATE ${include_dir})
target_link_libraries(spy PRIVATE
	fmt::fmt-header-only
	argparse::argparse
	magic_enum::magic_enum
	unofficial::concurrentqueue::concurrentqueue
)

# test
add_subdirectory(test)