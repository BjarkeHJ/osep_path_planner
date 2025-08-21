/* 

ROS2 Node for Path Planning

*/
#include <rclcpp/rclcpp.hpp>


class PlannerNode : public rclcpp::Node {
public:
    PlannerNode();
    
private:

};

PlannerNode::PlannerNode() : Node("PlannerNode") {

}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlannerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}