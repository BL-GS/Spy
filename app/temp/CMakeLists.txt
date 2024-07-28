CMAKE_MINIMUM_REQUIRED(VERSION 3.20)
PROJECT(temp)

ADD_EXECUTABLE(${PROJECT_NAME} temp.cpp)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE
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