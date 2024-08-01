CMAKE_MINIMUM_REQUIRED(VERSION 3.20)
SET(CMAKE_POLICY_DEFAULT_CMP0126 NEW)

PROJECT(spy)

SET(CMAKE_C_STANDARD 17)
SET(CMAKE_CXX_STANDARD 20)

IF (CMAKE_BUILD_TYPE STREQUAL Release)
	ADD_DEFINITIONS(-DNDEBUG)
ELSE()
	SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fsanitize=address")
ENDIF()

# Hardware Architecture --------------------------------------------------

INCLUDE(CMakeArch.cmake)

# Libraries --------------------------------------------------------------
FIND_PACKAGE(fmt	 	CONFIG REQUIRED)
FIND_PACKAGE(spdlog		CONFIG REQUIRED)
FIND_PACKAGE(magic_enum CONFIG REQUIRED)
FIND_PACKAGE(argparse 	CONFIG REQUIRED)
FIND_PACKAGE(unofficial-perfetto CONFIG REQUIRED)
FIND_PACKAGE(unofficial-concurrentqueue CONFIG REQUIRED)

ADD_DEFINITIONS(-DUNOFFICIAL_CONCURRENTQUEUE)

# Submodules -------------------------------------------------------------

# -- utilities
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/util)
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/perf)

# -- storage
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/storage)

# -- orchestration
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/orchestration)

# -- model
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/model)

# -- backend
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/backend)

# -- apps
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/app/cli)
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/app/temp)
