CMAKE_MINIMUM_REQUIRED(VERSION 3.20)
PROJECT(SpyModel)

FILE(GLOB_RECURSE source_files ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Combine all libraries
ADD_LIBRARY(${PROJECT_NAME} STATIC ${source_files})
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC
    # Internal libraries
    SpyUtil
    SpyOrchestration
    # External libraries
    magic_enum::magic_enum
)