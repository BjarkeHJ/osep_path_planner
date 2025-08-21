#pragma once

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/exceptions.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <algorithm>

class ESDF2dCostMapNode : public rclcpp::Node {
public:
  // --- Constructor ---
  ESDF2dCostMapNode();

private:
  // --- Callback Methods ---
  void esdf_callback(const sensor_msgs::msg::PointCloud2::SharedPtr esdf_msg);

  // --- Parameters and Computed Values ---
  int local_grid_size_;
  int global_grid_size_;
  double resolution_;
  double free_center_radius_;
  double local_map_size_;
  double global_map_size_;
  double local_half_size_;
  double global_half_size_;
  double safety_distance_;
  double safety_distance_min_;
  double safety_distance_max_;
  std::string frame_id_;
  std::string costmap_topic_;

  // --- TF2 Buffer and Listener ---
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // --- ROS Publishers and Subscribers ---
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr esdf_sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr global_map_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr local_map_pub_;

  // --- Global Map ---
  nav_msgs::msg::OccupancyGrid global_map_;
};
