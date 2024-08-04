cmake_minimum_required(VERSION 3.20)
project(temp)

add_executable(${PROJECT_NAME} temp.cpp)
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