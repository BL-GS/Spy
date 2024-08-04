cmake_minimum_required(VERSION 3.20)
set(CMAKE_POLICY_DEFAULT_CMP0126 NEW)

project(spy)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)

IF (CMAKE_BUILD_TYPE STREQUAL Release)
	add_definitions(-DNDEBUG)
else()
	set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fsanitize=address")
endif()

# Hardware Architecture --------------------------------------------------

include(CMakeArch.cmake)

# Options
option(SPY_APP			"Enable compiling apps"					ON)

option(SPY_TEST			"Enable compiling test suites" 			OFF)

option(SPY_BENCHMARK 	"Enable compiling benchmark suites"		OFF)

# Libraries --------------------------------------------------------------
find_package(fmt	 	CONFIG REQUIRED)
find_package(spdlog		CONFIG REQUIRED)
find_package(magic_enum CONFIG REQUIRED)
find_package(argparse 	CONFIG REQUIRED)
find_package(unofficial-perfetto CONFIG REQUIRED)
find_package(unofficial-concurrentqueue CONFIG REQUIRED)

add_definitions(-DUNOFFICIAL_CONCURRENTQUEUE)

# Submodules -------------------------------------------------------------

# -- utilities
add_subdirectory(${PROJECT_SOURCE_DIR}/util)
add_subdirectory(${PROJECT_SOURCE_DIR}/perf)

# -- storage
add_subdirectory(${PROJECT_SOURCE_DIR}/storage)

# -- orchestration
add_subdirectory(${PROJECT_SOURCE_DIR}/orchestration)

# -- model
add_subdirectory(${PROJECT_SOURCE_DIR}/model)

# -- backend
add_subdirectory(${PROJECT_SOURCE_DIR}/backend)

# -- apps
if(${SPY_APP})
	add_subdirectory(${PROJECT_SOURCE_DIR}/app/cli)
	add_subdirectory(${PROJECT_SOURCE_DIR}/app/temp)
endif()

# -- test
if(${SPY_TEST})
	enable_testing()
	add_subdirectory(${PROJECT_SOURCE_DIR}/test)
endif()

# -- benchmark
if(${SPY_BENCHMARK})
	add_subdirectory(${PROJECT_SOURCE_DIR}/benchmark)
endif()
