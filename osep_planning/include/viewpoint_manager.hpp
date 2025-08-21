#ifndef VIEWPOINT_MANAGER_HPP_
#define VIEWPOINT_MANAGER_HPP_

#include <iostream>
#include <chrono>
#include <pcl/common/common.h>
#include <Eigen/Core>

#define RUN_STEP(fn) \
    do { \
        bool ok = (fn)(); \
        running = ok; \
        if (!ok) return false; \
    } while (0)

struct ViewpointConfig {

};

struct Viewpoint {
    Eigen::Vector3f position;
    Eigen::Quaternionf orientation;

    int corresp_vertex_id = -1;
    float score = 0.0f;
    bool in_path = false;
    bool visited = false;
    bool invalid = false;
};

struct ViewpointData {
    pcl::PointCloud<pcl::PointXYZ>::Ptr gskel;
    std::vector<Viewpoint> global_vpts;

};    


class ViewpointManager {
public:
    ViewpointManager(const ViewpointConfig &cfg);
    bool viewpoint_manager_run();
    pcl::PointCloud<pcl::PointXYZ>& input_skeleton() { return *VD.gskel; }


private:
    /* Functions */
    bool viewpoint_sampling();
    bool viewpoint_filtering();
    bool viewpoint_visibility_graph();

    /* Helper */
    std::vector<Viewpoint> generate_viewpoint(int id);

    /* Params */
    ViewpointConfig cfg_;
    bool running;

    /* Data */
    ViewpointData VD;

    /* Utils */

};

#endif