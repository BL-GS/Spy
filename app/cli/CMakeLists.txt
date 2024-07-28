CMAKE_MINIMUM_REQUIRED(VERSION 3.20)
PROJECT(cli)

FILE(GLOB_RECURSE source_files ${PROJECT_SOURCE_DIR}/src/*.cpp)
ADD_EXECUTABLE(${PROJECT_NAME} ${source_files})

TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)
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