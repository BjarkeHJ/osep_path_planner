/* 

ROS 2 Node: ROSA Point Computation and Incremental Skeleton 

*/

#include "rosa.hpp"

#include <mutex>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>


class SkeletonExtractionNode : public rclcpp::Node {
public:
    SkeletonExtractionNode();
    
private:
    /* Functions */
    void pcd_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr pcd_msg);
    void process_tick();
    
    /* ROS2 Sub/Pub/Timer */
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_sub_;
    // SOME ODOMETRY SUBSCRIPTION
    // SOME CURRENT SKELETON PUBLISH
    rclcpp::TimerBase::SharedPtr tick_timer_;

    /* Params */
    std::string topic_prefix_;
    int tick_ms_; 

    /* Utils */
    std::mutex latest_mutex_;
    
    /* Data */
    sensor_msgs::msg::PointCloud2::ConstSharedPtr latest_msg_;

};

SkeletonExtractionNode::SkeletonExtractionNode() : Node("SkeletonExtractionNode") {
    // DECLARE LAUNCH FILE PARAMETERS HERE
    topic_prefix_ = declare_parameter<std::string>("topic_prefix", "/osep");
    tick_ms_ = declare_parameter<int>("tick_ms", 50);

    // Initialize objects (rosa, gskel etc...) and parse parameters to their constructors

    auto qos = rclcpp::SensorDataQoS(); 
    qos.keep_last(1);
    pcd_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(topic_prefix_+"/lidar_scan", 
                                                                        qos,
                                                                        std::bind(&SkeletonExtractionNode::pcd_callback,
                                                                        this,
                                                                        std::placeholders::_1));
    
    tick_timer_ = create_wall_timer(std::chrono::milliseconds(50),
                                    std::bind(&SkeletonExtractionNode::process_tick, this));

    RCLCPP_INFO(this->get_logger(), "SkeletonExtractionNode Initialized");
}

void SkeletonExtractionNode::pcd_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr pcd_msg) {
    std::scoped_lock lk(latest_mutex_);
    latest_msg_ = std::move(pcd_msg);
}

void SkeletonExtractionNode::process_tick() {
    /* Processing runs every */
    sensor_msgs::msg::PointCloud2::ConstSharedPtr msg;
    {
        std::scoped_lock lk(latest_mutex_);
        msg = latest_msg_;
        latest_msg_.reset();
    }
    if (!msg) return;
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*msg, cloud);
    // Process data with ROSA CLASS (rosa->process()) return final vertices

    // Lookup transform for "msg"

    // Parse transform vector and quaternion (Eigen) to global skeleton module (gskel->increment(tf,vertices)) return global skeleton

    // Publish global skeleton as coordinates (or pcd??)
}



int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SkeletonExtractionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}