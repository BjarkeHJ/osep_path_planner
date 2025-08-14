cmake_minimum_required(VERSION 3.8)
project(osep_core)

find_package(ament_cmake REQUIRED)
find_package(path_planning REQUIRED)
find_package(data_publisher REQUIRED)
find_package(rosa REQUIRED)

install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME})

ament_package()
