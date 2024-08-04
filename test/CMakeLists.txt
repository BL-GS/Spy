cmake_minimum_required(VERSION 3.20)
project(SpyTest)

find_package(GTest CONFIG REQUIRED)

# Project setup
macro(add_spy_test TEST_NAME)
    add_library(${PROJECT_NAME} STATIC ${PROJECT_SOURCE_DIR}/src/${TEST_NAME}.cpp)
    target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        # Internal library
        SpyUtil
        SpyPerf
        SpyOrchestration
        SpyStorage
        SpyModel
        SpyBackend
        # External libraries
        GTest::gtest 
		GTest::gtest_main 
		GTest::gmock 
		GTest::gmock_main
    )
endmacro()