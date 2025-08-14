#pragma once

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <eigen3/Eigen/Dense>
#include <tf2/utils.h>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <unordered_map>

// Define a struct for A* nodes
struct PlannerNode {
    int x, y;
    float cost;
    bool operator>(const PlannerNode &other) const {
        return cost > other.cost;
    }
};

class PathInterpolator : public rclcpp::Node {
public:
    PathInterpolator();

private:
    static constexpr int obstacle_threshold_ = 50;

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr waypoints_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr viewpoints_adjusted_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr raw_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr smoothed_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr ground_truth_trajectory_pub_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    nav_msgs::msg::OccupancyGrid::SharedPtr costmap_;
    nav_msgs::msg::Path adjusted_waypoints_;
    std::string frame_id_;
    double interpolation_distance_;
    rclcpp::TimerBase::SharedPtr ground_truth_timer_;
    nav_msgs::msg::Path ground_truth_trajectory_;
    bool path_invalid_flag_ = false;
    double extra_safety_distance_;

    void costmapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void waypointsCallback(const nav_msgs::msg::Path::SharedPtr msg);
    geometry_msgs::msg::PoseStamped getCurrentPosition();
    void updateGroundTruthTrajectory();
    std::pair<geometry_msgs::msg::PoseStamped, bool> adjustWaypointForCollision(
        const geometry_msgs::msg::PoseStamped &waypoint, float distance, float resolution, int max_attempts);
    tf2::Quaternion interpolateYaw(
        const geometry_msgs::msg::Pose &start_pose,
        const geometry_msgs::msg::Pose &goal_pose,
        float t);
    void planAndPublishPath();
    std::vector<geometry_msgs::msg::PoseStamped> planPath(
        const geometry_msgs::msg::PoseStamped &start,
        const geometry_msgs::msg::PoseStamped &goal);
    std::vector<geometry_msgs::msg::PoseStamped> downsamplePath(
        const std::vector<geometry_msgs::msg::PoseStamped> &path, double min_distance);
    std::vector<geometry_msgs::msg::PoseStamped> smoothPath(
        const std::vector<geometry_msgs::msg::PoseStamped> &path, double interpolation_distance);
};
