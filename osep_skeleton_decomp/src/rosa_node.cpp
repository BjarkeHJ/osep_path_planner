/* 

ROS 2 Node: ROSA Point Computation

*/

#include "rosa.hpp"
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>


class RosaNode : public rclcpp::Node {
public:
    RosaNode();
    
private:
    /* Functions */
    void pcd_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr pcd_msg);
    void process_tick();
    void publish_vertices(Eigen::MatrixXf& skelver, const std_msgs::msg::Header& src_header);
    
    /* ROS2 Sub/Pub/Timer */
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_pub_;
    rclcpp::TimerBase::SharedPtr tick_timer_;
    // SOME ODOMETRY SUBSCRIPTION

    /* Params */
    std::string topic_prefix_;
    int tick_ms_; 

    /* Utils */
    std::mutex latest_mutex_;
    std::unique_ptr<Rosa> rosa_;

    /* Data */
    sensor_msgs::msg::PointCloud2::ConstSharedPtr latest_msg_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr skelver_cloud;
    RosaConfig rosa_cfg;
};

RosaNode::RosaNode() : Node("RosaNode") {
    /* LAUNCH FILE PARAMETER DECLARATIONS */
    // MISC
    topic_prefix_ = declare_parameter<std::string>("topic_prefix", "/osep");
    tick_ms_ = declare_parameter<int>("tick_ms", 200);
    // ROSA
    rosa_cfg.max_points = declare_parameter<int>("rosa_max_points", 1000);
    rosa_cfg.min_points = declare_parameter<int>("rosa_min_points", 300);
    rosa_cfg.pts_dist_lim = declare_parameter<float>("rosa_point_dist_lim", 75.0f);
    rosa_cfg.ne_knn = declare_parameter<int>("rosa_ne_knn", 20);
    rosa_cfg.nb_knn = declare_parameter<int>("rosa_nb_knn", 10);
    rosa_cfg.max_proj_range = declare_parameter<float>("rosa_max_projection_range", 10.0f);
    rosa_cfg.niter_drosa = declare_parameter<int>("rosa_niter_drosa", 6);
    rosa_cfg.niter_dcrosa = declare_parameter<int>("rosa_niter_dcrosa", 5);
    rosa_cfg.niter_smooth = declare_parameter<int>("rosa_niter_smooth", 3);
    rosa_cfg.alpha_recenter = declare_parameter<float>("rosa_recenter_alpha", 0.3f);
    rosa_cfg.radius_smooth = declare_parameter<float>("rosa_smooth_radius", 5.0f);

    /* OBJECT INITIALIZATION */
    rosa_ = std::make_unique<Rosa>(rosa_cfg);

    /* ROS2 */
    auto sub_qos = rclcpp::SensorDataQoS(); 
    sub_qos.keep_last(1);
    pcd_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/isaac/lidar/raw/pointcloud", 
                                                                        sub_qos,
                                                                        std::bind(&RosaNode::pcd_callback,
                                                                        this,
                                                                        std::placeholders::_1));
    
    auto pub_qos = rclcpp::QoS(rclcpp::KeepLast(10));
    pcd_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/test_pointcloud", pub_qos);
                                                                        
    tick_timer_ = create_wall_timer(std::chrono::milliseconds(tick_ms_),
                                    std::bind(&RosaNode::process_tick, this));

    /* Initialize Datastructures */
    skelver_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>); 

    RCLCPP_INFO(this->get_logger(), "RosaNode Initialized");
}

void RosaNode::pcd_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr pcd_msg) {
    std::scoped_lock lk(latest_mutex_);
    latest_msg_ = std::move(pcd_msg);
}

void RosaNode::publish_vertices(Eigen::MatrixXf& skelver, const std_msgs::msg::Header& src_header) {
    if (skelver.rows() == 0) return;
    skelver_cloud->clear();
    skelver_cloud->points.resize(skelver.rows());
    skelver_cloud->width = skelver.rows();
    skelver_cloud->height = 1;
    skelver_cloud->is_dense = true;

    for (int i=0; i<skelver.rows(); ++i) {
        auto& p = skelver_cloud->points[i];
        p.x = skelver(i,0);        
        p.y = skelver(i,1);        
        p.z = skelver(i,2);        
    }

    sensor_msgs::msg::PointCloud2 skelver_msg;
    pcl::toROSMsg(*skelver_cloud, skelver_msg);
    
    skelver_msg.header = src_header; // set original frame_id and stamp...
    if (skelver_msg.header.frame_id.empty()) skelver_msg.header.frame_id = "lidar";
    if (skelver_msg.header.stamp.sec == 0 && skelver_msg.header.stamp.nanosec == 0) {
        skelver_msg.header.stamp = this->get_clock()->now();
    }
    
    pcd_pub_->publish(skelver_msg);
}

void RosaNode::process_tick() {
    sensor_msgs::msg::PointCloud2::ConstSharedPtr msg;
    {
        std::scoped_lock lk(latest_mutex_);
        msg = latest_msg_;
        latest_msg_.reset();
    }
    if (!msg) return;

    pcl::fromROSMsg(*msg, rosa_->input_cloud());

    std::cout << "stamp: "
          << msg->header.stamp.sec << "."
          << std::setw(9) << std::setfill('0') << msg->header.stamp.nanosec
          << "  frame: " << msg->header.frame_id << std::endl;

    if (rosa_->rosa_run()) {
        Eigen::MatrixXf& local_vertices = rosa_->output_vertices();
        publish_vertices(local_vertices, msg->header);
    }

    // Lookup transform for "msg"

    // Parse transform vector and quaternion (Eigen) to global skeleton module (gskel->increment(tf,vertices)) return global skeleton

    // Publish global skeleton as coordinates (or pcd??)
}



int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RosaNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}