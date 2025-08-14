/* 

ROtational Symmetry Axis (ROSA) Skeletonization ROS2 Node

TODO:
- Constructor instead of init() function for setting up pubs/subs


*/


#include "rosa_main.hpp"

#include <chrono>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <pcl_conversions/pcl_conversions.h>

class SkeletonExtractionNode : public rclcpp::Node {
public:
    SkeletonExtractionNode() : Node("skeleton_extraction_node") {
        RCLCPP_INFO(this->get_logger(), "Skeleton Extraction Node Constructed");
    }

    void init();
    void pcd_callback(const sensor_msgs::msg::PointCloud2::SharedPtr pcd_msg);
    void set_transform();
    void run();
    void publish_vertices();

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr vertex_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
    rclcpp::TimerBase::SharedPtr run_timer_;
    rclcpp::TimerBase::SharedPtr vertex_pub_timer_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

private:
    std::string topic_prefix = "/osep";

    /* Utils */
    std::shared_ptr<SkelEx> skel_ex;

    /* Params */
    bool run_flag = false;
    int run_timer_ms = 50;
    int vertex_pub_timer_ms = 50;

    /* Data */
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_;
    geometry_msgs::msg::TransformStamped curr_tf;

    // std::string global_frame_id = "World";
    std::string global_frame_id = "odom";
    // std::string local_frame_id = "lidar_frame";
    std::string local_frame_id = "lidar";
};


void SkeletonExtractionNode::init() {
    RCLCPP_INFO(this->get_logger(), "Initializing Modules and Data Structures...");
    
    /* Subscriber, Publishers, Timers, etc... */
    pcd_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(topic_prefix+"/lidar_scan", 10, std::bind(&SkeletonExtractionNode::pcd_callback, this, std::placeholders::_1));
    vertex_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(topic_prefix+"/local_vertices", 10);
    cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(topic_prefix+"/local_points", 10);
    
    run_timer_ = this->create_wall_timer(std::chrono::milliseconds(run_timer_ms), std::bind(&SkeletonExtractionNode::run, this));
    vertex_pub_timer_ = this->create_wall_timer(std::chrono::milliseconds(vertex_pub_timer_ms), std::bind(&SkeletonExtractionNode::publish_vertices, this));

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    /* Params */
    // Stuff from launch file (ToDo)...

    /* Data */
    pcd_.reset(new pcl::PointCloud<pcl::PointXYZ>);


    /* Modules */
    skel_ex = std::make_shared<SkelEx>(shared_from_this());
    skel_ex->init();
}

void SkeletonExtractionNode::pcd_callback(const sensor_msgs::msg::PointCloud2::SharedPtr pcd_msg) {
    if (run_flag) return; // Currently processing...

    if (pcd_msg->data.empty()) {
        RCLCPP_INFO(this->get_logger(), "Received empty point cloud...");
        return;
    }
    pcl::fromROSMsg(*pcd_msg, *pcd_);
    skel_ex->SS.pts_ = pcd_;

    try {
        // curr_tf = tf_buffer_->lookupTransform("World", "lidar_frame", pcd_msg->header.stamp, rclcpp::Duration::from_seconds(0.1));
        // curr_tf = tf_buffer_->lookupTransform("World", "lidar_frame", tf2::TimePointZero);
        // curr_tf = tf_buffer_->lookupTransform(global_frame_id, local_frame_id, tf2::TimePointZero);
        curr_tf = tf_buffer_->lookupTransform(global_frame_id, local_frame_id, pcd_msg->header.stamp);
        set_transform();
    }
    catch (const tf2::TransformException &ex) {
        RCLCPP_ERROR(this->get_logger(), "Transform Lookup Failed: %s", ex.what());
        return;
    }

    run_flag = true;
}

void SkeletonExtractionNode::set_transform() {
    Eigen::Quaterniond q(curr_tf.transform.rotation.w,
        curr_tf.transform.rotation.x,
        curr_tf.transform.rotation.y,
        curr_tf.transform.rotation.z);
    Eigen::Matrix3d R = q.toRotationMatrix();
    Eigen::Vector3d t(curr_tf.transform.translation.x,
        curr_tf.transform.translation.y,
        curr_tf.transform.translation.z);
    skel_ex->tf_rot = R;
    skel_ex->tf_trans = t;
}

void SkeletonExtractionNode::run() {
    if (run_flag) {
        
        run_flag = false;
        skel_ex->main();

        sensor_msgs::msg::PointCloud2 cloud_in;
        sensor_msgs::msg::PointCloud2 cloud_out;
        pcl::toROSMsg(*skel_ex->SS.pts_, cloud_in);

        cloud_in.header.frame_id = local_frame_id;
        cloud_in.header.stamp = curr_tf.header.stamp;
        // cloud_in.header.stamp = this->now();

        try {
            // Transform the point cloud
            tf2::doTransform(cloud_in, cloud_out, curr_tf);

            // Pass through filter to filter points above gnd_threshold

            // Now cloud_out is in the target_frame
            cloud_out.header.frame_id = global_frame_id;
            cloud_pub_->publish(cloud_out);
        }
        catch (tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "Transform failed: %s", ex.what());
        }
    }
}

void SkeletonExtractionNode::publish_vertices() {
    if (skel_ex->SS.vertices_ && !skel_ex->SS.vertices_->empty()) {
        sensor_msgs::msg::PointCloud2 vertex_msg;
        pcl::toROSMsg(*skel_ex->SS.vertices_, vertex_msg);
        // vertex_msg.header.frame_id = local_frame_id;
        vertex_msg.header.frame_id = global_frame_id;
        vertex_msg.header.stamp = curr_tf.header.stamp;
        // vertex_msg.header.stamp = this->now();
        vertex_pub_->publish(vertex_msg);
    }
    else RCLCPP_INFO(this->get_logger(), "WARNING: Waiting for first vertex set...");
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SkeletonExtractionNode>();
    node->init(); // Initialize Modules etc...

    // Spin the node
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}