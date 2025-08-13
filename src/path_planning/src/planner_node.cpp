/*

Path Planner Node

*/

#include "path_planning/planner_main.hpp"
// #include "planner_main.hpp"

#include <chrono>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Core>

class PlannerNode : public rclcpp::Node {
public:
    PlannerNode() : Node("path_planner_node") {
        RCLCPP_INFO(this->get_logger(), "Skeleton Guided Path Planner Node Constructed");
    }

    void init();
    void pcd_callback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg);
    void vertex_callback(const sensor_msgs::msg::PointCloud2::SharedPtr vertex_msg);
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr odom_msg);

    void publish_branches();
    void publish_viewpoints();
    void publish_traced_path();
    void publish_seen_voxels();

    void publish_gskel();
    void publish_path();
    void init_path();
    void drone_tracking();
    void adjusted_viewpoints_callback(const nav_msgs::msg::Path::SharedPtr adjusted_vpts);

    void apply_viewpoint_adjustments();

    void run();

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr vertex_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr gskel_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr adj_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr branch_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr viewpoint_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr viewpoint_connection_pub_;

    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr adjusted_vpts_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr traced_path_pub_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr seen_voxels_pub_;
    
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr prelim_ver_pub_;

    rclcpp::TimerBase::SharedPtr run_timer_;
    rclcpp::TimerBase::SharedPtr traced_timer_;

private:
    std::string topic_prefix = "/osep";

    std::shared_ptr<PathPlanner> planner;

    bool update_skeleton_flag = false;
    bool planner_flag = false;
    bool path_init = false;
    bool vpts_adjusted = false;

    bool waiting_for_adjusted = false; 

    int run_cnt;

    int run_timer_ms = 100;
    int traced_ms = 100;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices;

    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;

    // std::string global_frame_id = "World";
    std::string global_frame_id = "odom";
};

void PlannerNode::init() {
    RCLCPP_INFO(this->get_logger(), "Initializing Modules and Data Structures...");

    /* Subscriber, Publishers, Timers, etc... */
    pcd_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(topic_prefix+"/local_points", 10, std::bind(&PlannerNode::pcd_callback, this, std::placeholders::_1));
    vertex_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(topic_prefix+"/local_vertices", 10, std::bind(&PlannerNode::vertex_callback, this, std::placeholders::_1));
    gskel_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(topic_prefix+"/global_skeleton", 10);
    adj_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(topic_prefix+"/adjacency_graph", 10);
    cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(topic_prefix+"/global_points", 10);

    branch_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(topic_prefix+"/branches", 10);

    path_pub_ = this->create_publisher<nav_msgs::msg::Path>(topic_prefix+"/viewpoints", 10);
    traced_path_pub_ = this->create_publisher<nav_msgs::msg::Path>(topic_prefix+"/traced_viewpoints", 10);
    viewpoint_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(topic_prefix+"/all_viewpoints", 10);
    viewpoint_connection_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(topic_prefix+"/viewpoint_connections", 10);

    seen_voxels_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(topic_prefix+"/seen_voxels", 10); // When a vpt is popped published the seen voxels 
    adjusted_vpts_sub_ = this->create_subscription<nav_msgs::msg::Path>("/planner/viewpoints_adjusted", 10, std::bind(&PlannerNode::adjusted_viewpoints_callback, this, std::placeholders::_1)); 

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("/isaac/odom", 10, std::bind(&PlannerNode::odom_callback, this, std::placeholders::_1));

    prelim_ver_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(topic_prefix+"/prelim_vertices", 10);

    run_timer_ = this->create_wall_timer(std::chrono::milliseconds(run_timer_ms), std::bind(&PlannerNode::run, this));
    traced_timer_ = this->create_wall_timer(std::chrono::milliseconds(traced_ms), std::bind(&PlannerNode::publish_traced_path, this));

    /* Params */
    // Stuff from launch file (ToDo)...

    /* Data */
    vertices.reset(new pcl::PointCloud<pcl::PointXYZ>);
    cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

    /* Modules */
    planner = std::make_shared<PathPlanner>(shared_from_this());
    planner->init();

    run_cnt = 0;

    RCLCPP_INFO(this->get_logger(), "Initialization Done...");
}

void PlannerNode::pcd_callback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
    if (!cloud_msg ||cloud_msg->data.empty()) {
        return;
    }
    pcl::fromROSMsg(*cloud_msg, *planner->local_pts);
    planner->global_cloud_handler();

    sensor_msgs::msg::PointCloud2 global_msg;
    pcl::toROSMsg(*planner->GS.global_pts, global_msg);
    global_msg.header.frame_id = global_frame_id;
    global_msg.header.stamp = cloud_msg->header.stamp;
    cloud_pub_->publish(global_msg);
}

void PlannerNode::vertex_callback(const sensor_msgs::msg::PointCloud2::SharedPtr vertex_msg) {
    if (!vertex_msg || vertex_msg->data.empty()) {
        return;
    }
    pcl::fromROSMsg(*vertex_msg, *vertices);
    planner->local_vertices = vertices;
    update_skeleton_flag = true;
}

void PlannerNode::odom_callback(const nav_msgs::msg::Odometry::SharedPtr odom_msg) {
    if (!odom_msg) {
        return;
    }
    position(0) = odom_msg->pose.pose.position.x;
    position(1) = odom_msg->pose.pose.position.y;
    position(2) = odom_msg->pose.pose.position.z;
    
    orientation.w() = odom_msg->pose.pose.orientation.w;
    orientation.x() = odom_msg->pose.pose.orientation.x;
    orientation.y() = odom_msg->pose.pose.orientation.y;
    orientation.z() = odom_msg->pose.pose.orientation.z;
}

void PlannerNode::publish_branches() {
    if (planner->GS.global_vertices_cloud->points.empty() || planner->GS.branches.empty()) return;
    // Build an RGB cloud
    pcl::PointCloud<pcl::PointXYZRGB> cloud_rgb;
    cloud_rgb.header.frame_id = global_frame_id;     // your fixed frame
    cloud_rgb.height = 1;

    const size_t N = planner->GS.branches.size();
    for (size_t bi = 0; bi < N; ++bi) {
        // compute a unique hue for this branch
        float h = float(bi) / float(N);
        // --- HSV → RGB (v=1, s=1) ---
        int   hi = int(h * 6) % 6;
        float f  = h * 6 - int(h * 6);
        float q  = 1.0f - f;
        float t  = f;
        uint8_t r, g, b;
        switch (hi) {
        case 0: r = 255; g = uint8_t(t * 255); b =   0; break;
        case 1: r = uint8_t(q * 255); g = 255; b =   0; break;
        case 2: r =   0; g = 255; b = uint8_t(t * 255); break;
        case 3: r =   0; g = uint8_t(q * 255); b = 255; break;
        case 4: r = uint8_t(t * 255); g =   0; b = 255; break;
        case 5: r = 255; g =   0; b = uint8_t(q * 255); break;
        }
        // -----------------------------

        for (int idx : planner->GS.branches[bi]) {
        if (idx < 0 || idx >= (int)planner->GS.global_vertices_cloud->points.size()) {
            RCLCPP_WARN(get_logger(),
            "Branch %zu has invalid index %d", bi, idx);
            continue;
        }
        const auto &pt = planner->GS.global_vertices_cloud->points[idx];
        pcl::PointXYZRGB p;
        p.x = pt.x; p.y = pt.y; p.z = pt.z;
        p.r = r;   p.g = g;   p.b = b;
        cloud_rgb.push_back(p);
        }
    }

    // Convert to ROS message and publish
    sensor_msgs::msg::PointCloud2 out;
    pcl::toROSMsg(cloud_rgb, out);
    out.header.stamp = now();
    branch_pub_->publish(out);
}

void PlannerNode::publish_traced_path() {
    if (planner->GP.traced_path.empty()) return;

    const auto& vpts = planner->GP.traced_path;
    if (!vpts.empty()) {
        nav_msgs::msg::Path path_msg;
        path_msg.header.frame_id = global_frame_id;
        path_msg.header.stamp = now();

        for (const auto& vp : vpts) {
            geometry_msgs::msg::PoseStamped pose_msg;
            pose_msg.header = path_msg.header;

            pose_msg.pose.position.x = vp.position.x();
            pose_msg.pose.position.y = vp.position.y();
            pose_msg.pose.position.z = vp.position.z();

            pose_msg.pose.orientation.x = vp.orientation.x();
            pose_msg.pose.orientation.y = vp.orientation.y();
            pose_msg.pose.orientation.z = vp.orientation.z();
            pose_msg.pose.orientation.w = vp.orientation.w();

            path_msg.poses.push_back(pose_msg);
        }
        traced_path_pub_->publish(path_msg);
    }
}

void PlannerNode::publish_seen_voxels() {
    if (planner->GS.global_seen_cloud->points.empty()) return;

    sensor_msgs::msg::PointCloud2 seen_msg;
    pcl::toROSMsg(*planner->GS.global_seen_cloud, seen_msg);
    seen_msg.header.frame_id = global_frame_id;
    seen_msg.header.stamp = now();
    seen_voxels_pub_->publish(seen_msg);
}

void PlannerNode::publish_gskel() {
    if (planner->GS.global_vertices_cloud && !planner->GS.global_vertices_cloud->empty()) {
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*planner->GS.global_vertices_cloud, output);
        output.header.frame_id = global_frame_id;
        output.header.stamp = now();
        gskel_pub_->publish(output);

        visualization_msgs::msg::Marker lines;
        lines.header.frame_id = global_frame_id;
        lines.header.stamp = this->get_clock()->now();
        lines.type = visualization_msgs::msg::Marker::LINE_LIST;
        lines.action = visualization_msgs::msg::Marker::ADD;
        lines.pose.orientation.w = 1.0;
        lines.scale.x = 0.05;
        lines.color.r = 0.0;
        lines.color.g = 0.0;
        lines.color.b = 0.1;
        lines.color.a = 1.0;

        std::set<std::pair<int, int>> published_edges;
        geometry_msgs::msg::Point p1, p2;
        for (int i = 0; i < (int)planner->GS.global_adj.size(); ++i) {
            const auto& neighbors = planner->GS.global_adj[i];
            for (int j : neighbors) {
                // Make sure we only publish each edge once
                if (i < j) {
                    p1.x = planner->GS.global_vertices_cloud->points[i].x;
                    p1.y = planner->GS.global_vertices_cloud->points[i].y;
                    p1.z = planner->GS.global_vertices_cloud->points[i].z;

                    p2.x = planner->GS.global_vertices_cloud->points[j].x;
                    p2.y = planner->GS.global_vertices_cloud->points[j].y;
                    p2.z = planner->GS.global_vertices_cloud->points[j].z;

                    lines.points.push_back(p1);
                    lines.points.push_back(p2);

                    published_edges.insert({i, j});
                }
            }
        }
        adj_pub_->publish(lines);
    }

    if (!planner->GS.prelim_vertices.empty()) {
        sensor_msgs::msg::PointCloud2 prelim_msg;
        pcl::PointCloud<pcl::PointXYZ>::Ptr prever_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointXYZ pt;
        for (auto& ver : planner->GS.prelim_vertices) {
            pt.x = ver.position.x();
            pt.y = ver.position.y();
            pt.z = ver.position.z();
            prever_cloud->points.push_back(pt);
        }
        pcl::toROSMsg(*prever_cloud, prelim_msg);
        prelim_msg.header.frame_id = global_frame_id;
        prelim_msg.header.stamp = now();
        prelim_ver_pub_->publish(prelim_msg);
    }
}

void PlannerNode::publish_viewpoints() {
    // Publish all generated viewpoints
    if (!planner->GP.global_vpts.empty()) {
        geometry_msgs::msg::PoseArray gvps_msg;
        gvps_msg.header.frame_id = global_frame_id;
        gvps_msg.header.stamp = now();
    
        // for (const auto& vp : planner->GP.all_vpts) {
        for (const auto& vp : planner->GP.global_vpts) {
            geometry_msgs::msg::Pose vp_pose;
            vp_pose.position.x = vp.position.x();
            vp_pose.position.y = vp.position.y();
            vp_pose.position.z = vp.position.z();
            vp_pose.orientation.x = vp.orientation.x();
            vp_pose.orientation.y = vp.orientation.y();
            vp_pose.orientation.z = vp.orientation.z();
            vp_pose.orientation.w = vp.orientation.w();
            gvps_msg.poses.push_back(vp_pose);
        }
        viewpoint_pub_->publish(gvps_msg);    
    }

    // Publish inter-viewpoint adjacency
    const auto& vpts = planner->GP.global_vpts;
    visualization_msgs::msg::Marker conns;
    conns.header.frame_id = global_frame_id;
    conns.header.stamp = this->get_clock()->now();
    conns.type  = visualization_msgs::msg::Marker::LINE_LIST;
    conns.action= visualization_msgs::msg::Marker::ADD;

    conns.pose.orientation.w = 1.0;
    conns.scale.x = 0.1;
    conns.color.r = 1.0;
    conns.color.g = 0.0;
    conns.color.b = 0.0;
    conns.color.a = 1.0;
    
    geometry_msgs::msg::Point p1, p2;
    // for each edge, push two points (start, end)
    for (const auto& vp : vpts) {
      // vp.adj is std::vector<Viewpoint*>
      for (auto* nbr : vp.adj) {
        // avoid drawing each undirected edge twice
        if (&vp < nbr) {
          geometry_msgs::msg::Point p1, p2;
          p1.x = vp.position.x();
          p1.y = vp.position.y();
          p1.z = vp.position.z();
          p2.x = nbr->position.x();
          p2.y = nbr->position.y();
          p2.z = nbr->position.z();
          conns.points.push_back(p1);
          conns.points.push_back(p2);
        }
      }
    }

    viewpoint_connection_pub_->publish(conns);
}

void PlannerNode::publish_path() {
    const auto& vpts = planner->GP.local_path;
    if (!vpts.empty()) {
        nav_msgs::msg::Path path_msg;
        path_msg.header.frame_id = global_frame_id;
        path_msg.header.stamp = now();

        for (const auto& vp : vpts) {
            geometry_msgs::msg::PoseStamped pose_msg;
            pose_msg.header = path_msg.header;

            pose_msg.pose.position.x = vp->position.x();
            pose_msg.pose.position.y = vp->position.y();
            pose_msg.pose.position.z = vp->position.z();

            pose_msg.pose.orientation.x = vp->orientation.x();
            pose_msg.pose.orientation.y = vp->orientation.y();
            pose_msg.pose.orientation.z = vp->orientation.z();
            pose_msg.pose.orientation.w = vp->orientation.w();

            path_msg.poses.push_back(pose_msg);
        }
        path_pub_->publish(path_msg);
    }
}

void PlannerNode::init_path() {
    // Guiding the drone towards the structure
    // Viewpoint* first_vp;
    Viewpoint* second_vp = new Viewpoint();
    Viewpoint* third_vp = new Viewpoint();
    Viewpoint* fourth_vp = new Viewpoint();

    // first_vp->position = Eigen::Vector3d(0.0, 0.0, 50.0);
    // first_vp->orientation = Eigen::Quaterniond::Identity();

    second_vp->position = Eigen::Vector3d(0.0, 0.0, 120.0);
    // second_vp->position = Eigen::Vector3d(0.0, 0.0, 140.0);
    second_vp->orientation = Eigen::Quaterniond::Identity();

    third_vp->position = Eigen::Vector3d(100.0, 0.0, 120.0);
    // third_vp->position = Eigen::Vector3d(100.0, 0.0, 140.0);
    third_vp->orientation = Eigen::Quaterniond::Identity();

    fourth_vp->position = Eigen::Vector3d(180.0, 0.0, 120.0);
    // fourth_vp->position = Eigen::Vector3d(180.0, 0.0, 140.0);
    fourth_vp->orientation = Eigen::Quaterniond::Identity();

    // planner->GP.local_path.push_back(first_vp);
    planner->GP.local_path.push_back(second_vp);
    planner->GP.local_path.push_back(third_vp);
    planner->GP.local_path.push_back(fourth_vp);
}

void PlannerNode::drone_tracking() {
    if (planner->GP.adjusted_path.empty()) return;

    const double dist_check_th = 2.0;

    for (int i=0; i<(int)planner->GP.adjusted_path.size(); ++i) {
        Viewpoint target = planner->GP.adjusted_path[i];
        // Viewpoint* target = planner->GP.local_path[i];
        double distance_to_drone = (target.position - position).norm();
        // double distance_to_drone = (target->position - position).norm();
        if (distance_to_drone < dist_check_th) {
            // Mark the corresponding viewpoint as visited...
            Viewpoint* master_vp = planner->GP.local_path[i];
            master_vp->visited = true;
            master_vp->in_path = false;
            // target->visited = true;
            // target->in_path = false;
            // Update seen voxels based on viewpoint visisted
            planner->update_seen_cloud(master_vp);
            // planner->update_seen_cloud(target);
            // Delete from paths
            planner->GP.traced_path.push_back(planner->GP.adjusted_path[i]);
            // planner->GP.traced_path.push_back(*planner->GP.local_path[i]);
            // planner->GP.adjusted_path.erase(planner->GP.adjusted_path.begin(), planner->GP.adjusted_path.begin() + (i+1));
            // planner->GP.local_path.erase(planner->GP.local_path.begin(), planner->GP.local_path.begin() + (i+1));

            // Remove only i from paths
            planner->GP.adjusted_path.erase(planner->GP.adjusted_path.begin() + i);
            planner->GP.local_path.erase(planner->GP.local_path.begin() + i);

            RCLCPP_INFO(this->get_logger(), "Reached Viewpoint: Deleting %d viewpoints from path", (i+1));
            break;
        }
    }
}

void PlannerNode::adjusted_viewpoints_callback(const nav_msgs::msg::Path::SharedPtr adjusted_vpts) {
    if (!adjusted_vpts) return;

    planner->GP.adjusted_path.clear();
    planner->GP.adjusted_path.reserve(adjusted_vpts->poses.size());

    for (const auto& ps : adjusted_vpts->poses) {
        Viewpoint vp;

        if (ps.header.frame_id.empty()) vp.invalid = true;

        vp.position = Eigen::Vector3d(
                    ps.pose.position.x,
                    ps.pose.position.y,
                    ps.pose.position.z
        );
        vp.orientation = Eigen::Quaterniond(
                        ps.pose.orientation.w,
                        ps.pose.orientation.x,
                        ps.pose.orientation.y,
                        ps.pose.orientation.z
        );

        planner->GP.adjusted_path.push_back(vp);
    }
    vpts_adjusted = true;
}

void PlannerNode::apply_viewpoint_adjustments() {
    if (!vpts_adjusted) return;
    auto &gp = planner->GP;
    auto &adjusted = gp.adjusted_path;
    auto &local    = gp.local_path;
    auto &gvp      = gp.global_vpts;

    if (adjusted.size() != local.size()) return;

    for (int i = (int)local.size() - 1; i >= 0; --i) {
        if (adjusted[i].invalid) {
        Viewpoint* dead = local[i];
        for (auto it = gvp.begin(); it != gvp.end(); ++it) {
            if (&*it == dead) { gvp.erase(it); break; }
        }
        local.erase(local.begin() + i);
        }
        else {
        local[i]->position    = adjusted[i].position;
        local[i]->orientation = adjusted[i].orientation;
        }
  }

  // 3) cleanup
  vpts_adjusted = false;
}

void PlannerNode::run() {
    if (!planner_flag) {
        /* Initial Flight to Structure (Predefined) */

        if (!path_init && planner->GP.local_path.empty()) {
            init_path(); // Set once
            path_init = true; 
            RCLCPP_INFO(this->get_logger(), "Initial Path Set - Following until structure detected!");     
        }

        if (!planner->local_vertices->empty()) {
            RCLCPP_INFO(this->get_logger(), "Recieved first vertices - Starting Planning!");
            for (auto vp_ptr : planner->GP.local_path) {
                delete vp_ptr;
            }              
            planner->GP.local_path.clear();
            planner->GP.adjusted_path.clear();
            planner_flag = true;
        }
        
        drone_tracking();
        publish_path();
        
        return;
    }

    if (update_skeleton_flag) {
        planner->update_skeleton();
        update_skeleton_flag = false;
        publish_branches();
    }

    publish_gskel();
    publish_viewpoints();
    publish_seen_voxels();
    
    // planner->pose = {position, orientation};
    // planner->plan_path();
    // publish_path();

    // if (run_cnt >= 20) {
        
    if (planner->GS.global_vertices.empty()) return;

    if (!waiting_for_adjusted) {
        drone_tracking();
        planner->pose = {position, orientation};
        planner->plan_path();
        publish_path();
        if (!planner->GP.local_path.empty()) {
            waiting_for_adjusted = true;
        }
    }
    else if (vpts_adjusted) {
        apply_viewpoint_adjustments();
        drone_tracking();
        vpts_adjusted = false;
        waiting_for_adjusted = false;
    }
    
    // }
    // else run_cnt++;
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlannerNode>();
    node->init(); // Initialize Modules etc...

    // Spin the node
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}