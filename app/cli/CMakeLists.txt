cmake_minimum_required(VERSION 3.20)
project(cli)

file(GLOB_RECURSE source_files ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_executable(${PROJECT_NAME} ${source_files})

target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} PRIVATE
	# System library
	${spy_system_libraries}
	# Internal library
	SpyUtil
	SpyPerf
	SpyOrchestration
	SpyStorage
	SpyModel
	SpyBackend
)