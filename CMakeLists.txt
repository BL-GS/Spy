CMAKE_MINIMUM_REQUIRED(VERSION 3.20)
set(CMAKE_POLICY_DEFAULT_CMP0126 NEW)

PROJECT(spy)

SET(CMAKE_C_STANDARD 17)
SET(CMAKE_CXX_STANDARD 20)

IF (CMAKE_BUILD_TYPE STREQUAL Release)
	ADD_DEFINITIONS(-DNDEBUG)
ENDIF()

# Hardware Architecture --------------------------------------------------

INCLUDE(CMakeArch.cmake)

# Libraries --------------------------------------------------------------
IF (${CMAKE_HOST_UNIX})
	SET(spy_system_libraries uring)
ENDIF ()

FIND_PACKAGE(fmt	 	CONFIG REQUIRED)
FIND_PACKAGE(magic_enum CONFIG REQUIRED)
FIND_PACKAGE(argparse 	CONFIG REQUIRED)
FIND_PACKAGE(unofficial-concurrentqueue CONFIG REQUIRED)

# Internal libraries -----------------------------------------------------

SET(spy_include_dir ${PROJECT_SOURCE_DIR}/include)
ADD_LIBRARY(spy_interface INTERFACE)
TARGET_INCLUDE_DIRECTORIES(spy_interface INTERFACE ${spy_include_dir})
TARGET_LINK_LIBRARIES(spy_interface INTERFACE
		# External library
		fmt::fmt-header-only
		argparse::argparse
		magic_enum::magic_enum
		unofficial::concurrentqueue::concurrentqueue
)

SET(spy_internal_libraries ${spy_internal_libraries} spy_interface)

# Backends ---------------------------------------------------------------
OPTION(SPY_BACKEND_CPU "Enable backend: CPU"  ON)
OPTION(SPY_BACKEND_GPU "Enable backend: GPU"  OFF)

IF (${SPY_BACKEND_CPU})
	ADD_SUBDIRECTORY(src/cpu)
	SET(spy_internal_libraries ${spy_internal_libraries} spy_cpu)
ENDIF ()

IF (${SPY_BACKEND_GPU})
	ADD_SUBDIRECTORY(src/gpu)
	SET(spy_internal_libraries ${spy_internal_libraries} spy_gpu)
ENDIF ()

# Main project ----------------------------------------------------------
ADD_EXECUTABLE(spy spy.cpp)
TARGET_INCLUDE_DIRECTORIES(spy PRIVATE ${spy_include_dir})
TARGET_LINK_LIBRARIES(spy PRIVATE
	# System library
	${spy_system_libraries}
	# External library
	fmt::fmt-header-only
	argparse::argparse
	magic_enum::magic_enum
	unofficial::concurrentqueue::concurrentqueue
	# Internal library
	${spy_internal_libraries}
)

ADD_EXECUTABLE(spytest test.cpp)
TARGET_INCLUDE_DIRECTORIES(spytest PRIVATE ${spy_include_dir})
TARGET_LINK_LIBRARIES(spytest PRIVATE
	# System library
	${spy_system_libraries}
	# External library
	fmt::fmt-header-only
	argparse::argparse
	magic_enum::magic_enum
	unofficial::concurrentqueue::concurrentqueue
	# Internal library
	${spy_internal_libraries}
)

# Test -----------------------------------------------------------------
ADD_SUBDIRECTORY(test)