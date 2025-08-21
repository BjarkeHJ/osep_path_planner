/* 

ROS2 Node for Viewpoint managing

*/
#include <rclcpp/rclcpp.hpp>

class ViewpointNode : public rclcpp::Node {
public:
    ViewpointNode();
    
private:

};

ViewpointNode::ViewpointNode() : Node("ViewpointManagerNode") {

}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ViewpointNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}