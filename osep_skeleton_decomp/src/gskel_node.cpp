/* 

ROS 2 Node: Global Skeleton
TODO: CMakeLists to spawn two nodes...!

*/

#include "gskel.hpp"
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>

class GSkelNode : public rclcpp::Node {
public:
    GSkelNode();

    /* ROS2 TF */
    private:
    /* Functions */
    void vertex_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr pcd_msg);
    void process_tick();
    /* ROS2 Sub/Pub/Timer */
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr vers_sub_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    /* Params */
    std::string topic_prefix_;
    std::string global_frame_id_;
    int tick_ms_;
    
    /* Utils */
    std::mutex latest_mutex_;
    std::unique_ptr<GSkel> gskel_;
    
    /* Data */
    GSkelConfig gskel_cfg;
    sensor_msgs::msg::PointCloud2::ConstSharedPtr latest_msg_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr latest_cands_;
};

GSkelNode::GSkelNode() : Node("GSkelNode") {
    /* LAUNCH FILE PARAMETER DECLARATIONS */

    // MISC
    topic_prefix_ = declare_parameter<std::string>("topic_prefix", "/osep");
    global_frame_id_ = declare_parameter<std::string>("global_frame_id", "/Odom");
    // GSKEL
    gskel_cfg.fuse_dist_th = declare_parameter<float>("gskel_fuse_dist_th", 3.0f);
    gskel_cfg.fuse_conf_th = declare_parameter<float>("gskel_fuse_conf_th", 0.1f);
    gskel_cfg.lkf_pn = declare_parameter<float>("gskel_lkf_pn", 0.0001);
    gskel_cfg.lkf_mn = declare_parameter<float>("gskel_lkf_mn", 0.1);

    /* OBJECT INITIALIZATION */
    gskel_ = std::make_unique<GSkel>(gskel_cfg);

    /* ROS2 */
    auto sub_qos = rclcpp::SensorDataQoS();
    sub_qos.keep_last(1);
    vers_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/test_pointcloud",
                                                                            sub_qos,
                                                                            std::bind(&GSkelNode::vertex_callback,
                                                                            this,
                                                                            std::placeholders::_1));
    
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
                                                                            
    /* INITIALIZE DATA STRUCTURES */
    latest_cands_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    latest_cands_->points.reserve(1000);

    RCLCPP_INFO(this->get_logger(), "GSkelNode Initialized");
}

void GSkelNode::vertex_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr vertex_msg) {
    if (!vertex_msg) return;
    std::scoped_lock lk(latest_mutex_);
    latest_msg_ = std::move(vertex_msg);
}

void GSkelNode::process_tick() {
    sensor_msgs::msg::PointCloud2::ConstSharedPtr msg;
    {
        std::scoped_lock lk(latest_mutex_);
        msg = latest_msg_;
        latest_msg_.reset();
    }
    
    if (!msg) return;
    // get transform and transform point
    geometry_msgs::msg::TransformStamped T;
    try {
        T = tf_buffer_->lookupTransform(global_frame_id_,
                                        msg->header.frame_id,
                                        msg->header.stamp,
                                        rclcpp::Duration::from_seconds(0.2));
    }
    catch (const tf2::TransformException& ex) {
        RCLCPP_WARN(this->get_logger(), "TF Lookup failed: %s", ex.what());
        return;
    }

    sensor_msgs::msg::PointCloud2 gmsg;
    tf2::doTransform(*msg, gmsg, T);
    pcl::fromROSMsg(gmsg, *latest_cands_);
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GSkelNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}