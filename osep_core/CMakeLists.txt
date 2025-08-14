cmake_minimum_required(VERSION 3.8)
project(osep_core)

find_package(ament_cmake REQUIRED)
find_package(osep_data_publisher REQUIRED)
find_package(osep_skeleton_decomp REQUIRED)
find_package(osep_viewpoint_manager REQUIRED)
find_package(osep_planning REQUIRED)

install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME})

ament_package()
