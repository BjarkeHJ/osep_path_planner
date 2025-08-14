/* 

ROS 2 Node: ROSA Point Computation and Incremental Skeleton 

*/

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>

class SkeletonExtractionNode : public rclcpp::Node {
public:
    SkeletonExtractionNode();
    ~SkeletonExtractionNode();

private:
rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_sub_;
// SOME ODOMETRY SUBSCRIPTION
// SOME CURRENT SKELETON PUBLISH

std::string topic_prefix = "/osep";

};

SkeletonExtractionNode::SkeletonExtractionNode() : Node("SkeletonExtractionNode") {
    auto qos = rclcpp::SensorDataQoS(); 
    qos.keep_last(1);
    pcd_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(topic_prefix+"/lidar_scan", qos, );

    RCLCPP_INFO(this->get_logger(), "SkeletonExtractionNode Initialized");
}


