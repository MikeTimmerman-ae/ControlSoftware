cmake_minimum_required(VERSION 3.0.0)
project(ControlSoftware VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(ControlSoftware main.cpp)

add_subdirectory(libraries/eigen)
add_subdirectory(libraries/qpOASES)

target_include_directories(${PROJECT_NAME}
    PUBLIC libraries/eigen
    PUBLIC libraries/qpOASES/include
)

target_link_directories(${PROJECT_NAME}
    PUBLIC libraries/eigen
    PRIVATE libraries/qpOASES
)

target_link_libraries(${PROJECT_NAME} eigen qpOASES)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
