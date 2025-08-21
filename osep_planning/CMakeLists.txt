cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0074 NEW)
project(osep_planning)

# Set compile constraints
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define executable
set(EXEC_PLAN osep_planer)
set(EXEC_VIEW osep_viewpoint_manager)

# ROS2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Dependencies list
set(dependencies
    rclcpp
    pcl_conversions
    tf2_ros
    tf2_sensor_msgs
    sensor_msgs
    geometry_msgs)

# Other libraries
find_package(Eigen3 REQUIRED)
find_package(PCL 1.12 REQUIRED)

add_executable(${EXEC_PLAN}
                src/planner_node.cpp)

add_executable(${EXEC_VIEW}
                src/viewpoint_manager_node.cpp)
        
foreach(tgt IN ITEMS ${EXEC_PLAN} ${EXEC_VIEW})

  target_include_directories(${tgt} PUBLIC
                              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                              $<INSTALL_INTERFACE:include>
                              ${PCL_INCLUDE_DIRS})

  target_link_libraries(${tgt} 
                        ${PCL_LIBRARIES} 
                        Eigen3::Eigen)

  ament_target_dependencies(${tgt} 
                            ${dependencies})

  target_compile_features(${tgt} 
                          PUBLIC cxx_std_17)

endforeach()

install(TARGETS 
        ${EXEC_PLAN}
        ${EXEC_VIEW} 
        DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
        DESTINATION include)

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