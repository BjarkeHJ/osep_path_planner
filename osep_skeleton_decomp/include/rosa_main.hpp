#ifndef ROSA_MAIN_HPP_
#define ROSA_MAIN_HPP_

#include "extract_tools.hpp"

#include <algorithm>
#include <random>

#include <rclcpp/rclcpp.hpp>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/features/normal_3d.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

struct Vector3dCompare // lexicographic ordering: return true if v1 is ordered BEFORE v2...
{
    bool operator()(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) const {
        if (v1(0) != v2(0)) return v1(0) < v2(0); // Return if x components differ (False if x1 > x2)
        if (v1(1) != v2(1)) return v1(1) < v2(1); // Only if x1 = x2 (False if y1 > y2)
        return v1(2) < v2(2); // Only if y1 = y2
    }
};
 
struct SkeletonStructure {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pts_;
    pcl::PointCloud<pcl::Normal>::Ptr normals_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices_; // Final local vertices

    Eigen::MatrixXd pts_matrix;
    Eigen::MatrixXd nrs_matrix;

    std::vector<std::vector<int>> neighs;
    std::vector<std::vector<int>> surf_neighs;

    Eigen::MatrixXd skelver; // Local skeleton vertices
    Eigen::MatrixXd corresp; // Point vertex correspondency
    Eigen::MatrixXi Adj;
};


class SkelEx {
public: 
    SkelEx(rclcpp::Node::SharedPtr node); // Constructor with ROS2 node parsed

    /* Pipeline Functions */
    void init();
    void main();
    void distance_filter();
    void normal_estimation();
    void similarity_neighbor_extraction();
    void drosa();
    void dcrosa();
    void vertex_sampling();
    void vertex_smooth();
    void vertex_recenter();
    void get_vertices();

    /* Helper Functions */
    double similarity_metric(pcl::PointXYZ &p1, pcl::Normal &v1, pcl::PointXYZ &p2, pcl::Normal &v2, double range_r);
    void rosa_init(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud<pcl::Normal>::Ptr &normals);
    Eigen::Matrix3d create_orthonormal_frame(Eigen::Vector3d &v);
    Eigen::MatrixXd compute_active_samples(int &idx, Eigen::Vector3d &p_cut, Eigen::Vector3d &v_cut);
    Eigen::Vector3d compute_symmetrynormal(Eigen::MatrixXd& local_normals);
    double symmnormal_variance(Eigen::Vector3d& symm_nor, Eigen::MatrixXd& local_normals);
    Eigen::Vector3d symmnormal_smooth(Eigen::MatrixXd &V, Eigen::MatrixXd &w);
    Eigen::Vector3d closest_projection_point(const Eigen::MatrixXd& P, const Eigen::MatrixXd& V);
    Eigen::Vector3d PCA(Eigen::MatrixXd &normals);
    int argmax_eigen(Eigen::MatrixXd &x);

    /* Utils */

    /* Data */
    SkeletonStructure SS;

    Eigen::Matrix3d tf_rot;
    Eigen::Vector3d tf_trans;

private:
    rclcpp::Node::SharedPtr node_;

    /* Params */
    double pts_dist_lim = 50;
    int max_points = 500; //Goal for downsampling
    int drosa_iter = 3; //Number of drosa iterations
    int dcrosa_iter = 2;
    
    int ne_KNN = 20; //Normal Estimation neighbors
    int k_KNN = 10; //Surface neighbors 
    double leaf_size_ds;
    double max_projection_range = 10.0; //Maximum projection range during ROSA position [m]
    double alpha = 0.3; //Recenter fuse coefficient
    double gnd_th = 50; // Threshold for ground vertices removal (assumption of starting in the sky...)

    /* Data */
    int pcd_size_;
    double norm_scale;
    Eigen::Vector4d centroid;
    double delta;

    Eigen::MatrixXd pset; //Pointset for ROSA computation
    Eigen::MatrixXd vset; //Symmetry-axis vector set for ROSA computation
    Eigen::MatrixXd vvar; //Symmetry-axis vector variance in local region
    pcl::PointCloud<pcl::PointXYZ>::Ptr pset_cloud;
};

#endif //SKEL_MAIN_HPP_