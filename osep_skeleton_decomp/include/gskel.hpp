#ifndef GSKEL_HPP_
#define GSKEL_HPP_

#include <iostream>
#include <chrono>
#include <pcl/common/common.h>
#include <Eigen/Core>

#define RUN_STEP(fn) \
    do { \
        bool ok = (fn)(); \
        running = ok; \
        std::cout << (ok ? "[SUCCESS] " : "[FAILED] ") << #fn << std::endl; \
        if (!ok) return false; \
    } while (0)

struct GSkelConfig {
    float fuse_dist_th;
    float fuse_conf_th;
    float lkf_pn;
    float lkf_mn;
};

struct Vertex {
    Eigen::Vector3f position;
    Eigen::Matrix3f covariance;
    int obs_coubt = 0;
};

struct GSkelData {
    pcl::PointCloud<pcl::PointXYZ>::Ptr new_cands;
    
    std::vector<Vertex> prelim_vers;
    std::vector<Vertex> global_vers;
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_vers_cloud;
    std::vector<int> new_vers_indxs;
    

    std::vector<int> joints;
    std::vector<int> leafs;
    std::vector<std::vector<int>> branches;
    std::vector<std::vector<int>> global_adj;

    size_t gskel_size;
};

class GSkel {
public:
    /* Public methods */
    explicit GSkel(const GSkelConfig& cfg);
    bool run_gskel();
    pcl::PointCloud<pcl::PointXYZ>& input_vertices() { return *GD.new_cands; }

private:
    /* Functions */
    bool increment_skeleton();

    /* Helper */

    /* Params */
    GSkelConfig cfg_;
    bool running;

    /* Data */
    GSkelData GD;

    /* Utils */

};


#endif // GSKEL_HPP_