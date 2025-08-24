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
#include <optional>


class ESDF2dCostMapNode : public rclcpp::Node {
public:
  // --- Constructor ---
  ESDF2dCostMapNode();

private:
  // Functions
  bool validate_pointcloud2_fields(const sensor_msgs::msg::PointCloud2& msg);
  pcl::PointCloud<pcl::PointXYZI>::Ptr convert_to_pcl_cloud(const sensor_msgs::msg::PointCloud2& msg);
  std::optional<geometry_msgs::msg::TransformStamped> get_transform_to_odom();
  nav_msgs::msg::OccupancyGrid create_local_map(const geometry_msgs::msg::TransformStamped& transform);
  void extract_local_from_global(nav_msgs::msg::OccupancyGrid& local_map);
  void overwrite_local_with_esdf(
    nav_msgs::msg::OccupancyGrid& local_map,
    const std::vector<float>& esdf_grid,
    const std::vector<bool>& esdf_mask
  );  
  void clear_local_center(nav_msgs::msg::OccupancyGrid& local_map);
  void merge_local_to_global(const nav_msgs::msg::OccupancyGrid& local_map);
  void publish_maps(const nav_msgs::msg::OccupancyGrid& local_map);
  void convert_cloud_to_esdf_grid(
    const sensor_msgs::msg::PointCloud2& msg,
    std::vector<float>& esdf_grid,
    std::vector<bool>& esdf_mask,
    int grid_width,
    int grid_height,
    float origin_x,
    float origin_y,
    float resolution
  );


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
