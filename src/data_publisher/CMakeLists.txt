cmake_minimum_required(VERSION 3.8)
project(data_publisher)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

cmake_policy(SET CMP0074 NEW)

set(EXEC_NAME data_publisher)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)

set(dependencies
    rclcpp
    sensor_msgs
    pcl_conversions)

find_package(PCL 1.12 REQUIRED)

add_executable(${EXEC_NAME}
              src/lidar_pub_node.cpp)

target_include_directories(${EXEC_NAME} PUBLIC
                          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                          $<INSTALL_INTERFACE:include>
                          ${PCL_INCLUDE_DIRS})

target_link_libraries(${EXEC_NAME} 
                      ${PCL_LIBRARIES})

ament_target_dependencies(${EXEC_NAME} ${dependencies})

install(TARGETS ${EXEC_NAME} DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()