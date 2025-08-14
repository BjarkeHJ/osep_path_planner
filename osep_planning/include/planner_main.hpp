#ifndef PLANNER_MAIN_
#define PLANNER_MAIN_

#include "lkf_vertex_fuse.hpp"

#include <algorithm>
#include <queue>
#include <rclcpp/rclcpp.hpp>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Core>
#include <random>

struct SkeletonVertex;
struct Viewpoint;

struct VoxelIndex {
    int x, y, z;
    bool operator==(const VoxelIndex &other) const {
        return std::tie(x, y, z) == std::tie(other.x, other.y, other.z);
    }

    bool operator<(const VoxelIndex &other) const {
        return std::tie(x, y, z) < std::tie(other.x, other.y, other.z);
    }
};

struct VoxelIndexHash {
    std::size_t operator()(const VoxelIndex &k) const {
        return std::hash<int>()(k.x) ^ std::hash<int>()(k.y << 1) ^ std::hash<int>()(k.z << 2);
    }
};

struct Edge
{
    int u, v; // Vertex indices of the edge
    double w; // Lenght / weight of the edge
    bool operator<(const Edge &other) const {
        return w < other.w;
    } // comparison of the weight of this edge with another
};

struct UnionFind {
    std::vector<int> parent;  // parent[i] tells you who the parent of node i is

    // Constructor: initially, every node is its own parent (disconnected)
    UnionFind(int n) : parent(n) {
        for (int i = 0; i < n; ++i) parent[i] = i;
    }

    // Find the "representative" of the component that x belongs to
    int find(int x) {
        // If x is not its own parent, follow the chain recursively
        if (parent[x] != x)
            parent[x] = find(parent[x]);  // Path compression for speed
        return parent[x];
    }

    // Try to merge the sets that x and y belong to
    bool unite(int x, int y) {
        int rx = find(x);  // root of x
        int ry = find(y);  // root of y
        if (rx == ry) return false;  // Already in the same set — adding this edge would create a cycle
        parent[ry] = rx;  // Union: make one root the parent of the other
        return true;
    }
};

struct DronePose {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
};

struct SkeletonVertex {
    Eigen::Vector3d position;
    Eigen::Matrix3d covariance;
    int obs_count = 0;
    int unconfirmed_check = 0;
    bool just_approved = false;
    bool conf_check = false;
    bool freeze = false;
    bool marked_for_deletion = false;

    int smooth_iters_left = 3;

    int type = -1; // "0: invalid", "1: leaf", "2: branch", "3: joint" 
    int prev_type = -1;

    bool updated = false;
    bool spawned_vpts = false;
    std::vector<Viewpoint*> assigned_vpts;
};

struct Viewpoint {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    std::vector<VoxelIndex> covered_voxels;
    int corresp_vertex_id;
    std::vector<Viewpoint*> adj;
    double score = 0.0f;
    bool in_path = false;
    bool visited = false;
    bool invalid = false;
};

struct GlobalSkeleton {
    std::unordered_set<VoxelIndex, VoxelIndexHash> voxels;
    std::unordered_map<VoxelIndex, int, VoxelIndexHash> voxel_point_count;
    std::unordered_set<VoxelIndex, VoxelIndexHash> global_seen_voxels;
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_seen_cloud;

    pcl::PointCloud<pcl::PointXYZ>::Ptr global_pts;
    
    std::vector<SkeletonVertex> prelim_vertices;
    std::vector<SkeletonVertex> global_vertices;
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_vertices_cloud; // For visualizing
    std::vector<int> new_vertex_indices;

    std::vector<int> joints;
    std::vector<int> leafs;

    std::vector<std::vector<int>> global_adj;
    std::vector<std::vector<int>> branches;

    int gskel_size;
};

struct GlobalPath {
    std::vector<int> vertex_nbs_id;
    Viewpoint start;

    std::list<Viewpoint> global_vpts;
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_vpts_cloud;
    pcl::KdTreeFLANN<pcl::PointXYZ> global_vpts_tree;

    std::vector<Viewpoint*> local_path; // Current local path being published
    std::vector<Viewpoint> adjusted_path;
    std::vector<Viewpoint> traced_path; // Add only when drone reaches the vpt

};

class PathPlanner {
public:
    PathPlanner(rclcpp::Node::SharedPtr node);

    void init();
    void plan_path();
    void update_skeleton();
        
    /* Occupancy and Coverage */
    void global_cloud_handler();
    void update_seen_cloud(Viewpoint *vp);

    /* Data */
    GlobalSkeleton GS;
    GlobalPath GP;
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_pts;
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_vertices;
    DronePose pose;

private:
    rclcpp::Node::SharedPtr node_;

    /* Updating Skeleton */
    void skeleton_increment();
    void graph_adj();
    void mst();
    void vertex_merge();
    void prune_branches();
    
    void extract_branches();

    void smooth_vertex_positions();
    void graph_decomp();
    void merge_into(int id_keep, int id_del);
    
    /* Waypoint Generation and PathPlanning*/
    void viewpoint_sampling();
    void viewpoint_filtering();
    void build_visibility_graph();
    void generate_path();
    void refine_path();

    void trigger_replan();

    std::vector<Viewpoint> generate_viewpoint(int id);
    std::vector<Viewpoint> vp_sample(const Eigen::Vector3d& origin, const std::vector<Eigen::Vector3d>& directions, std::vector<double> dists, int vertex_id);
    bool viewpoint_check(const Viewpoint& vp, pcl::KdTreeFLANN<pcl::PointXYZ>& voxel_tree);
    bool viewpoint_similarity(const Viewpoint& a, const Viewpoint& b);
    void score_viewpoint(Viewpoint *vp);
    bool corridor_obstructed(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2);
    double distance_to_free_space(const Eigen::Vector3d &p, Eigen::Vector3d& dir);
    std::vector<Viewpoint*> dfs_run(Viewpoint* start_vp, const int max_depth);
    double dfs_reward(Viewpoint* node, Viewpoint* nb);
    int find_branch_index(int id);

    /* Data */
    double mean_edge_dist = 1.0;
    double mean_dfs_time = 0.0;
    double mean_dfs_reward = 0.0;

    int dfs_plan_cnt = 0;
    pcl::KdTreeFLANN<pcl::PointXYZ> global_pts_kdtree;

    /* Params */
    // int max_obs_wo_conf = 3; // Maximum number of runs without passing conf check before discarding...
    int max_obs_wo_conf = 5; // Maximum number of runs without passing conf check before discarding...
    double fuse_dist_th = 3.0;
    double fuse_conf_th = 0.1;
    double kf_pn = 0.0001;
    double kf_mn = 0.1;
    
    double voxel_size = 1.5;
    double fov_h = 90;
    double fov_v = 60;
    
    double disp_dist = 12;
    double absolute_dist_th = 18;
    double min_view_dist = 4;
    double max_view_dist = 20;
    double safe_dist = 6;
    double corridor_radius = 2;
    
    double viewpoint_merge_dist = 3.0;
    double visibility_graph_radius = 20.0;
    double gnd_th = 60.0;
    
    const int MAX_HORIZON = 2;
    // const int DFS_MAX_DEPTH = 5;
    const int DFS_MAX_DEPTH = 7;
    // const int DFS_MAX_DEPTH = 10;
    const int BEAM_WIDTH = 3;
    const double REPLAN_THRESH = 0.0;

    const double LEAF_W = 20.0;
    // const double LEAF_W = 3.0;
    const double COVERAGE_W = 10.0;
    const double BRANCH_W = 5.0;
    // const double BRANCH_W = 1.5;
    const double REVISIT_W = 7.0;
    const double DIST_W = 1.0;
};

#endif //PLANNER_MAIN_