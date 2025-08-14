cmake_minimum_required(VERSION 3.10)
project(osep_core)

find_package(ament_cmake REQUIRED)
find_package(osep_planning REQUIRED)
find_package(osep_data_publisher REQUIRED)
find_package(osep_rosa REQUIRED)
find_package(osep_2d_map REQUIRED)

install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME})

ament_package()
