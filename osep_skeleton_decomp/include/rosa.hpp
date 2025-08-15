#ifndef ROSA_HPP_
#define ROSA_HPP_

#include <chrono>
#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>

struct RosaConfig {
    int max_points;
    int est_vertices;
    float pts_dist_lim; 
    int ne_knn;
    float vg_ds_size;
};

struct CloudData {
    pcl::PointCloud<pcl::PointXYZ>::Ptr orig_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pts_;
    pcl::PointCloud<pcl::Normal>::Ptr nrms_;
    size_t pcd_size_;
};

struct RosaResult {
    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices_;
};

class Rosa {
public:
    explicit Rosa(const RosaConfig& cfg);
    bool rosa_run();
    pcl::PointCloud<pcl::PointXYZ>& input_cloud() { return *CD.orig_; } // Return mutable ref so node can write into preallocated buffer

private:
    /* Functions */
    void preprocess();


    /* Params */
    RosaConfig cfg_;

    /* Utils */
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne_;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr ne_tree_{new pcl::search::KdTree<pcl::PointXYZ>};
    pcl::VoxelGrid<pcl::PointNormal> vgf_;

    /* Data */
    CloudData CD;
    RosaResult RR;

    pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_pt_;
    pcl::PointCloud<pcl::Normal>::Ptr tmp_nrm_;
    pcl::PointCloud<pcl::PointNormal>::Ptr tmp_pn_;
    pcl::PointCloud<pcl::PointNormal>::Ptr tmp_pn_ds_;
    

};

#endif // ROSA_HPP_