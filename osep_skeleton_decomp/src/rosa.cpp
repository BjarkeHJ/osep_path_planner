/*

Main algorithm for local ROSA Point computation

*/

#include <rosa.hpp>

Rosa::Rosa(const RosaConfig& cfg) : cfg_(cfg) {
    CD.orig_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    CD.pts_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    CD.nrms_.reset(new pcl::PointCloud<pcl::Normal>);
    CD.pts_->points.reserve(cfg.max_points);
    CD.nrms_->points.reserve(cfg.max_points);

    RR.vertices_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    RR.vertices_->points.reserve(cfg.est_vertices);

    tmp_pt_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    tmp_pt_->points.reserve(10000);
    tmp_nrm_.reset(new pcl::PointCloud<pcl::Normal>);
    tmp_nrm_->points.reserve(10000);
    tmp_pn_.reset(new pcl::PointCloud<pcl::PointNormal>);
    tmp_pn_->points.reserve(10000);
    tmp_pn_ds_.reset(new pcl::PointCloud<pcl::PointNormal>);
    tmp_pn_ds_->points.reserve(cfg_.max_points);
}

bool Rosa::rosa_run() {
    preprocess();

}

void Rosa::preprocess() {
    auto pp_ts = std::chrono::high_resolution_clock::now();
    CD.pts_->clear();
    CD.nrms_->clear();
    RR.vertices_->clear();

    if (CD.orig_->points.empty()) {
        return;
    }

    /* Distance Filtering */
    const float r2 = cfg_.pts_dist_lim * cfg_.pts_dist_lim;
    tmp_pt_->clear();
    tmp_pt_->points.resize(CD.orig_->points.size());
    size_t n = 0;

    const auto& src = CD.orig_->points;
    auto& dst = tmp_pt_->points;

    for (size_t i=0; i<src.size(); ++i) {
        const auto& p = src[i];
        const float d2 = p.x*p.x + p.y*p.y + p.z*p.z;
        if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) && d2 <= r2) {
            dst[n++] = p;
        }
    }
    dst.resize(n);
    CD.orig_->swap(*tmp_pt_);
    CD.orig_->width = CD.orig_->points.size();
    CD.orig_->height = 1;
    CD.orig_->is_dense = true;
    CD.pcd_size_ = CD.orig_->points.size();

    /* Normal Estimation */
    int pts_lim = static_cast<int>(std::floor(cfg_.max_points*0.8));
    if (CD.pcd_size_ < pts_lim) {
        tmp_pt_->clear();
        tmp_nrm_->clear();
        return;
    }

    ne_.setInputCloud(CD.orig_);
    ne_.setViewPoint(0.0, 0.0, 0.0);
    ne_.setSearchMethod(ne_tree_);
    ne_.setKSearch(cfg_.ne_knn);
    ne_.compute(*tmp_nrm_);

    /* Downsampling */
    tmp_pn_->clear();
    tmp_pn_->points.resize(tmp_pt_->points.size());
    for (size_t i=0; i<tmp_pn_->points.size(); ++i) {
        auto& q = tmp_pn_->points[i];
        const auto& p = tmp_pt_->points[i];
        const auto& n = tmp_nrm_->points[i];
        q.x = p.x;
        q.y = p.y;
        q.z = p.z;
        q.normal_x = n.normal_x;
        q.normal_y = n.normal_y;
        q.normal_z = n.normal_z;
    }
    tmp_pn_->width = tmp_pn_->points.size();
    tmp_pn_->height = 1;
    
    float leaf = estimate_leaf_from_bbox(*tmp_pt_, cfg_.max_points);
    for (int i=0; i<2; i++) {
        vgf_.setInputCloud(tmp_pn_);
        vgf_.setLeafSize(leaf, leaf, leaf);
        vgf_.filter(*tmp_pn_ds_);
                
        const size_t N = tmp_pn_ds_->points.size();
        if (N == 0) {
            leaf *= 0.5;
            continue; // go to loop-start with before swap
        }
    
        float ratio = tmp_pn_->points.size() / cfg_.max_points;
        if (ratio > 1.05) {
            leaf *= std::cbrt(ratio);
            tmp_pn_->swap(*tmp_pn_ds_);
        }
        else {
            break;
        }
    }
    
    const size_t M = tmp_pn_ds_->points.size();
    CD.pts_->clear();
    CD.pts_->resize(M);
    CD.nrms_->clear();
    CD.nrms_->resize(M);

    for (size_t i=0; i<tmp_pn_ds_->points.size(); ++i) {
        const auto& q = tmp_pn_ds_->points[i];
        CD.pts_->points[i] = pcl::PointXYZ(q.x, q.y, q.z);
        Eigen::Vector3f nn(q.normal_x, q.normal_y, q.normal_z);
        float L = nn.norm();
        if (L>1e-6) nn /= L; else nn = Eigen::Vector3f::Zero();
        CD.nrms_->points[i].normal_x = nn.x();
        CD.nrms_->points[i].normal_y = nn.y();
        CD.nrms_->points[i].normal_z = nn.z();
    }
    
    CD.pts_->width = CD.nrms_->width = tmp_pn_ds_->points.size();
    CD.pts_->height = CD.nrms_->height = 1;
    CD.pts_->is_dense = true;
    CD.nrms_->is_dense = true;
    
    tmp_pt_->clear();
    tmp_nrm_->clear();
    tmp_pn_->clear();
    tmp_pn_ds_->clear();

    auto pp_te = std::chrono::high_resolution_clock::now();
    auto pp_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(pp_te-pp_ts).count();
    std::cout << "[ROSA] Preprocess Time: " << pp_elapsed << std::endl;
}

static float estimate_leaf_from_bbox(const pcl::PointCloud<pcl::PointXYZ>& cloud, size_t target) {
    pcl::PointXYZ minp, maxp;
    pcl::getMinMax3D(cloud, minp, maxp);
    const float dx = std::max(1e-6f, maxp.x - minp.x);
    const float dy = std::max(1e-6f, maxp.y - minp.y);
    const float dz = std::max(1e-6f, maxp.z - minp.z);
    const float vol = dx * dy * dz;
    float leaf = std::cbrtf(vol / std::max<double>(1.0, target));
    return leaf;
}