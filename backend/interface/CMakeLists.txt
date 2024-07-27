CMAKE_MINIMUM_REQUIRED(VERSION 3.20)
PROJECT(SpyBackendInterface)

# Library
FIND_PACKAGE(magic_enum CONFIG REQUIRED)

# Project setup
ADD_LIBRARY(${PROJECT_NAME} INTERFACE)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} INTERFACE ${PROJECT_SOURCE_DIR}/include)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} INTERFACE
    # Internal libraries
    SpyUtil
    SpyOrchestration
    # External libraries
    magic_enum::magic_enum
)