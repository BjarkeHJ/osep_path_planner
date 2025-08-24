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
    int test;
};

struct Vertex {
    int vid = -1;
    std::vector<int> nb_ids;
    pcl::PointXYZ position;
    int type = 0;
    bool pos_update = false;
    bool type_update = false;
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
    size_t gskel_size;
    std::vector<Vertex> global_skel;
    std::vector<Viewpoint> global_vpts;
    std::vector<int> updated_vertices;
};    


class ViewpointManager {
public:
    ViewpointManager(const ViewpointConfig &cfg);
    bool viewpoint_run();
    std::vector<Vertex>& input_skeleton() { return VD.global_skel; }

private:
    /* Functions */
    bool viewpoint_sampling();
    bool viewpoint_filtering();
    bool viewpoint_visibility_graph();

    /* Helper */
    void fetch_updated_vertices();
    std::vector<Viewpoint> generate_viewpoint(int id);


    /* Params */
    ViewpointConfig cfg_;
    bool running;

    /* Data */
    ViewpointData VD;

    /* Utils */

};

#endif