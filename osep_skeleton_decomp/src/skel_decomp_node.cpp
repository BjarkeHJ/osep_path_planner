/* 

ROS 2 Node: ROSA Point Computation and Incremental Skeleton 

*/

#include "rosa.hpp"

#include <mutex>

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
    std::unique_ptr<Rosa> rosa_;
    std::mutex latest_mutex_;

    /* Data */
    sensor_msgs::msg::PointCloud2::ConstSharedPtr latest_msg_;
    RosaConfig rosa_cfg;

};

SkeletonExtractionNode::SkeletonExtractionNode() : Node("SkeletonExtractionNode") {
    // DECLARE LAUNCH FILE PARAMETERS HERE
    topic_prefix_ = declare_parameter<std::string>("topic_prefix", "/osep");
    tick_ms_ = declare_parameter<int>("tick_ms", 50);
    rosa_cfg.max_points = declare_parameter<int>("rosa_max_points", 500);
    rosa_cfg.est_vertices = declare_parameter<int>("rosa_est_vertice", 50);
    rosa_cfg.pts_dist_lim = declare_parameter<float>("rosa_point_dist_lim", 50);
    rosa_cfg.ne_knn = declare_parameter<int>("rosa_ne_knn", 20);
    rosa_cfg.vg_ds_size = declare_parameter<float>("rosa_vg_ds_size", 0.3);

    // Initialize objects (rosa, gskel etc...) and parse parameters to their constructors
    rosa_ = std::make_unique<Rosa>(rosa_cfg);

    auto qos = rclcpp::SensorDataQoS(); 
    qos.keep_last(1);
    pcd_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(topic_prefix_+"/lidar_scan", 
                                                                        qos,
                                                                        std::bind(&SkeletonExtractionNode::pcd_callback,
                                                                        this,
                                                                        std::placeholders::_1));
    
    tick_timer_ = create_wall_timer(std::chrono::milliseconds(tick_ms_),
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

    pcl::fromROSMsg(*msg, rosa_->input_cloud());
    // Throw if no correct message??

    if (rosa_->rosa_run()) {
        // TODO: Parse to incremental skeleton manager...
    }

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