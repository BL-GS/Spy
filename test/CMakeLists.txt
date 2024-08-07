cmake_minimum_required(VERSION 3.20)
project(SpyTest)

find_package(GTest CONFIG REQUIRED)

# Project setup
macro(add_spy_test TEST_NAME)
    add_executable(test_${TEST_NAME} ${PROJECT_SOURCE_DIR}/src/${TEST_NAME}.cpp)
    target_include_directories(test_${TEST_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(test_${TEST_NAME} PRIVATE
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

    add_test(AllTestsInMain test_${TEST_NAME})
endmacro()

add_spy_test(temp)