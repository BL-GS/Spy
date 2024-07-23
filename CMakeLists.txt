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
IF (${CMAKE_HOST_UNIX})
	SET(spy_system_libraries uring)
ENDIF ()

FIND_PACKAGE(fmt	 	CONFIG REQUIRED)
FIND_PACKAGE(spdlog		CONFIG REQUIRED)
FIND_PACKAGE(magic_enum CONFIG REQUIRED)
FIND_PACKAGE(argparse 	CONFIG REQUIRED)
FIND_PACKAGE(unofficial-concurrentqueue CONFIG REQUIRED)

ADD_DEFINITIONS(-DUNOFFICIAL_CONCURRENTQUEUE)

# Submodules -------------------------------------------------------------

# -- utilities
ADD_SUBDIRECTORY(util)
ADD_SUBDIRECTORY(perf)

# -- storage
ADD_SUBDIRECTORY(storage)

# -- orchestration
ADD_SUBDIRECTORY(orchestration)

# -- model
ADD_SUBDIRECTORY(model)

# -- backend
ADD_SUBDIRECTORY(backend)

# Main project ----------------------------------------------------------
ADD_EXECUTABLE(spy spy.cpp)
TARGET_INCLUDE_DIRECTORIES(spy PRIVATE ${PROJECT_SOURCE_DIR}/include)
TARGET_LINK_LIBRARIES(spy PRIVATE
	# System library
	${spy_system_libraries}
	# Internal library
	SpyUtil
	SpyOrchestration
	SpyBackend
	SpyModel

	# External library
	magic_enum::magic_enum
	unofficial::concurrentqueue::concurrentqueue
)

# Test -----------------------------------------------------------------
# ADD_SUBDIRECTORY(test)

ADD_EXECUTABLE(temp temp.cpp)
TARGET_INCLUDE_DIRECTORIES(temp PRIVATE ${PROJECT_SOURCE_DIR}/include)
TARGET_LINK_LIBRARIES(temp PRIVATE
	# System library
	${spy_system_libraries}
	# Internal library
	SpyUtil
	SpyPerf
	SpyOrchestration
	SpyStorage
	SpyModel

	# External library
	fmt::fmt-header-only
	magic_enum::magic_enum
	unofficial::concurrentqueue::concurrentqueue
)