/* 

ROS2 Node for Viewpoint managing

*/

#include "viewpoint_manager.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>

class ViewpointNode : public rclcpp::Node {
public:
    ViewpointNode();
    
private:
    /* Functions */

    /* ROS2 */
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr gskel_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr vp_pub;

    /* Params */
    std::string gskel_topic_;
    std::string vp_topic_;

    /* Utils */

    /* Data */

};

ViewpointNode::ViewpointNode() : Node("ViewpointManagerNode") {
    /* LAUNCH FILE PARAMETER DECLARATIONS */
    // MISC
    gskel_topic_ = declare_parameter<std::string>("gskel_topic", "/osep/gskel/global_skeleton_vertices");
    vp_topic_ = declare_parameter<std::string>("viewpoint_topic", "/osep/viewpoint_manager/viewpoints");
    // VIEWPOINT MANAGER
    

    /* OBJECT INITIALIZATION */

    /* ROS2 */

    /* INITIALIZE DATA STRUCTURES */

}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ViewpointNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}