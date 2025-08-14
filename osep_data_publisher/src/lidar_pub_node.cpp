/*

Node for publishing Real-Time Lidar data
Simple republisher to keep format...

*/

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>

class LidarPublisher : public rclcpp::Node {
public:
    LidarPublisher() : Node("lidar_publisher_node") {
        
        sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(pcd_sub_topic, 10, std::bind(&LidarPublisher::publishPointCloud, this, std::placeholders::_1));
        pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(pcd_pub_topic, 10);
    }

private:
    std::string topic_prefix = "/osep";
    std::string pcd_sub_topic = "/isaac/lidar/raw/pointcloud";
    std::string pcd_pub_topic = topic_prefix + "/lidar_scan";

    // std::string local_frame_id = "lidar_frame";
    std::string local_frame_id = "lidar";

    void publishPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr pcd_msg) {
        pcd_msg->header.frame_id = local_frame_id;
        // pcd_msg->header.stamp = this->now();
        pcd_msg->header.stamp = pcd_msg->header.stamp;
        size_t num_pts = pcd_msg->width * pcd_msg->height;
        pub_->publish(*pcd_msg);
        RCLCPP_INFO(this->get_logger(), "Published Point Cloud - Size: %zu", num_pts);
    }

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarPublisher>());
    rclcpp::shutdown();
    return 0;
  }