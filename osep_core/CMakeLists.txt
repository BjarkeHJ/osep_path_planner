cmake_minimum_required(VERSION 3.10)
project(osep_core)

find_package(ament_cmake REQUIRED)
find_package(osep_2d_map REQUIRED)
find_package(osep_skeleton_decomp REQUIRED)
find_package(osep_planning REQUIRED)

install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME})

ament_package()
