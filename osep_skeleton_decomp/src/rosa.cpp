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

    CD.surf_nbs.reserve(cfg.max_points);

    RR.vertices.reset(new pcl::PointCloud<pcl::PointXYZ>);
    RR.vertices->points.reserve(100);

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
    rosa_init();
    similarity_neighbor_extraction();
    drosa();
    dcrosa();

    return true;
}

void Rosa::preprocess() {
    auto pp_ts = std::chrono::high_resolution_clock::now();
    CD.pts_->clear();
    CD.nrms_->clear();
    RR.vertices->clear();

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
    ne_.setSearchMethod(kd_tree_);
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
            // Too aggressive?
            leaf *= 0.5;
            continue; // go to loop-start with before swap
        }
    
        float ratio = static_cast<float>(tmp_pn_->points.size()) / static_cast<float>(cfg_.max_points);
        if (ratio > 1.05) {
            leaf *= std::cbrt(ratio);
            tmp_pn_->swap(*tmp_pn_ds_);
        }
        else {
            break;
        }
    }

    leaf_size = leaf;

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

    CD.pcd_size_ = CD.pts_->points.size();

    auto pp_te = std::chrono::high_resolution_clock::now();
    auto pp_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(pp_te-pp_ts).count();
    std::cout << "[ROSA] Preprocess Time: " << pp_elapsed << std::endl;
}

void Rosa::rosa_init() {
    Eigen::Vector3f nv;
    Eigen::Matrix3f M;
    pset.resize(CD.pcd_size_, 3);
    vset.resize(CD.pcd_size_, 3);
    vvar.resize(CD.pcd_size_, 1);
    for (int i=0; i<CD.pcd_size_; ++i) {
        const auto m = CD.pts_->points[i].getVector3fMap();
        pset.row(i) = m.transpose();
        nv(0) = CD.nrms_->points[i].normal_x;
        nv(1) = CD.nrms_->points[i].normal_y;
        nv(2) = CD.nrms_->points[i].normal_z;
        M = create_orthonormal_frame(nv);
        vset.row(i) = M.col(0);
    }
}

void Rosa::similarity_neighbor_extraction() {
    CD.simi_nbs.clear();
    CD.simi_nbs.resize(CD.pcd_size_);

    kd_tree_->setInputCloud(CD.pts_);
    std::vector<int> nb_indxs;
    std::vector<float> nb_dists;
    nb_indxs.reserve(32);
    nb_dists.reserve(32);
    const float radius_r = 10.0f * leaf_size;
    const float th_sim = 0.1f * leaf_size;

    for (int i=0; i<CD.pcd_size_; ++i) {
        nb_indxs.clear();
        nb_dists.clear();

        const pcl::PointXYZ& P1 = CD.pts_->points[i];
        const pcl::Normal& N1 = CD.nrms_->points[i];

        if (!std::isfinite(N1.normal_x) || !std::isfinite(N1.normal_y) || !std::isfinite(N1.normal_z)) {
            CD.simi_nbs[i].clear();
            continue;
        }

        const Eigen::Map<const Eigen::Vector3f> p1(&P1.x);
        Eigen::Vector3f v1(N1.normal_x, N1.normal_y, N1.normal_z);
        v1.normalize();

        kd_tree_->radiusSearch(P1, radius_r, nb_indxs, nb_dists);

        std::vector<int> temp_neighs;
        temp_neighs.reserve(nb_indxs.size());

        for (int idx : nb_indxs) {
            if (idx == i) continue;
            const pcl::PointXYZ& P2 = CD.pts_->points[idx];
            const pcl::Normal& N2 = CD.nrms_->points[idx];
            
            if (!std::isfinite(N2.normal_x) || !std::isfinite(N2.normal_y) || !std::isfinite(N2.normal_z)) {
                continue;
            }

            const Eigen::Map<const Eigen::Vector3f> p2(&P2.x);
            Eigen::Vector3f v2(N2.normal_x, N2.normal_y, N2.normal_z);
            v2.normalize();
            const float w = similarity_metric(p1, v1, p2, v2, radius_r); // can take 6th arg scale (default 5.0)
            if (w > th_sim) {
                temp_neighs.push_back(idx);
            }
        }
        CD.simi_nbs[i].swap(temp_neighs);
    }
}

void Rosa::drosa() {
    CD.surf_nbs.clear();
    CD.surf_nbs.resize(CD.pcd_size_);
    std::vector<int> knn_idxs(cfg_.nb_knn);
    std::vector<float> knn_dists(cfg_.nb_knn);
    kd_tree_->setInputCloud(CD.pts_);
    for (int i=0; i<CD.pcd_size_; ++i) {
        kd_tree_->nearestKSearch(i, cfg_.nb_knn, knn_idxs, knn_dists);
        CD.surf_nbs[i] = knn_idxs;
    }   

    /* ROSA POINT ORIENTATION*/
    Eigen::Vector3f p, n, new_v;
    std::vector<int> active;
    Eigen::MatrixXf vnew = Eigen::MatrixXf::Zero(CD.pcd_size_, 3);
    
    for (int it=0; it<cfg_.niter_drosa; ++it) {
        for (int pidx=0; pidx<CD.pcd_size_; ++pidx) {
            p = pset.row(pidx);
            n = vset.row(pidx);
            active = compute_active_samples(pidx, p, n);
            
            if (!active.empty()) {
                new_v = compute_symmetrynormal(active);
                vnew.row(pidx) = new_v.transpose();
                vvar(pidx, 0) = symmnormal_variance(new_v, active);
            }
            else {
                vvar(pidx, 0) = 0.0f;
            }
        }
        vset = vnew;
        
        constexpr float eps = 1e-5;
        vvar = (vvar.array().square().square() + eps).inverse().matrix();
        
        // Smoothing...
        for (int p=0; p<CD.pcd_size_; ++p) {
            const auto& snb = CD.surf_nbs[p];
            if (snb.empty()) continue;
            
            Eigen::Matrix3f M = Eigen::Matrix3f::Zero();
            for (int j : snb) {
                const Eigen::Vector3f v = vset.row(j).transpose();
                const float w = vvar(j,0);
                M.noalias() += w * (v * v.transpose());
            }
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(M);
            vset.row(p) = es.eigenvectors().col(2).transpose();
            vset.row(p).normalize();
        }
    }
    
    /* ROSA POINT POSITION */
    float PLANAR_TH = 0.1;
    std::vector<Eigen::Vector3f> goodCenters;
    tmp_pt_->resize(CD.pcd_size_);
    std::vector<int> poorIdx;

    for (int pidx=0; pidx<CD.pcd_size_; ++pidx) {
        p = pset.row(pidx);
        n = vset.row(pidx);

        active = compute_active_samples(pidx, p, n);

        if (active.empty()) {
            poorIdx.push_back(pidx);
            continue;
        }

        Eigen::Vector3f evals = cov_eigs_from_normals(active);
        const float eps = 1e-6f;
        float planar_score = (evals(1) - evals(0)) / std::max(evals(2), eps);

        Eigen::Vector3f center_f;

        if (planar_score < PLANAR_TH) {
            Eigen::Vector3f mean_p = Eigen::Vector3f::Zero();
            Eigen::Vector3f mean_n = Eigen::Vector3f::Zero();
            for (int i : active) {
                mean_p += CD.pts_->points[i].getVector3fMap();
                const auto v = CD.nrms_->points[i].getNormalVector3fMap();
                const float s2 = v.squaredNorm();
                const float inv_len = (s2 > 1e-20f) ? (1.0f / std::sqrt(s2)) : 0.0f;
                const Eigen::Vector3f vn = v * inv_len;
                mean_n += vn;
            }
            const float inv_m = 1.0f / static_cast<float>(active.size());
            mean_p *= inv_m;
            mean_n *= inv_m;
            center_f = mean_p - mean_n * (0.5f * cfg_.max_proj_range);
        }
        else {
            center_f = closest_projection_point(active);
        }

        const bool finite = center_f.allFinite();
        const bool in_range = (center_f - p).norm() < cfg_.max_proj_range;
        if (finite && in_range) {
            pset.row(pidx) = center_f;
            const pcl::PointXYZ& qp = CD.pts_->points[pidx];
            tmp_pt_->points[pidx] = qp;
            goodCenters.push_back(center_f);
        }
        else {
            poorIdx.push_back(pidx);
        }
    }

    kd_tree_->setInputCloud(tmp_pt_);
    std::vector<int> pair_id(1);
    std::vector<float> pair_dist(1);
    for (int poor_id : poorIdx) {
        const pcl::PointXYZ& q = CD.pts_->points[poor_id];
        if (kd_tree_->nearestKSearch(q, 1, pair_id, pair_dist) > 0) {
            const int gid = pair_id[0];
            pset.row(poor_id) = goodCenters[gid].transpose();
        }
    }

    tmp_pt_->clear();
}

void Rosa::dcrosa() {
    constexpr float CONF_TH = 0.5;
    constexpr int MIN_NB = 3;
    Eigen::MatrixXf newpset(CD.pcd_size_, 3);

    for (int it=0; it<cfg_.niter_dcrosa; ++it) {
        for (int i=0; i<CD.pcd_size_; ++i) {
            const auto& nb = CD.simi_nbs[i];
            if (nb.empty()) {
                newpset.row(i) = pset.row(i);
                continue;
            }
            Eigen::Vector3f acc = Eigen::Vector3f::Zero();
            for (int j : nb) {
                acc += pset.row(j).transpose();
            }
            newpset.row(i) = (acc / static_cast<float>(nb.size())).transpose();
        }
        pset.swap(newpset);
    
        for (int i=0; i<CD.pcd_size_; ++i) {
            const auto& nb = CD.simi_nbs[i];
            newpset.row(i) = pset.row(i);
    
            if (static_cast<int>(nb.size()) < MIN_NB) continue;
    
            Eigen::Vector3f mean, dir;
            float conf;
            if (local_line_fit(CD.simi_nbs[i], mean, dir, conf, MIN_NB) && conf > CONF_TH) {
                const Eigen::Vector3f x = pset.row(i).transpose();
                const float t = dir.dot(x - mean);
                newpset.row(i) = (mean + t * dir).transpose();
            }
            else {
                newpset.row(i) = pset.row(i);
            }
        }
        pset.swap(newpset);
    }
}

void Rosa::vertex_sampling() {
    if (!pset_cloud) {
        pset_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    }

    if (pset_cloud->points.size() != static_cast<size_t>(CD.pcd_size_)) {
        pset_cloud->points.resize(CD.pcd_size_);
        pset_cloud->width = CD.pcd_size_;
        pset_cloud->height = 1;
        pset_cloud->is_dense = true;
    }

    for (int i=0; i<CD.pcd_size_; ++i) {
        pset_cloud->points[i].x = pset(i,0);
        pset_cloud->points[i].y = pset(i,1);
        pset_cloud->points[i].z = pset(i,2);
    }

    const float R = static_cast<float>(leaf_size);
    const float R2 = R * R;

    std::vector<char> covered(CD.pcd_size_, 0);
    std::vector<int> corresp(CD.pcd_size_, -1);
    std::vector<float> mind2(CD.pcd_size_, std::numeric_limits<float>::infinity());

    CD.skelver.resize(0,3);
    kd_tree_->setInputCloud(pset_cloud);
    
    std::vector<int> indxs;
    std::vector<float> d2s;
    int num_covered = CD.pcd_size_;

    for (int start=0; start<CD.pcd_size_ && num_covered>0; ) {
        int seed = -1;
        for (int i=0; i<CD.pcd_size_; ++i) {
            if (!covered[i]) {
                seed = i;
                start = i+1;
                break;
            }
        }
        if (seed == -1) break; // all covered

        const int vid = static_cast<int>(CD.skelver.rows());
        CD.skelver.conservativeResize(vid + 1, 3);
        CD.skelver.row(vid) = pset.row(seed);

        indxs.clear();
        d2s.clear();
        const pcl::PointXYZ& q = pset_cloud->points[seed];
        const int found = kd_tree_->radiusSearch(q, R, indxs, d2s);

        if (found <= 0) {
            covered[seed] = 1;
            num_covered -= 1;
            mind2[seed] = 0.0f;
            corresp[seed] = vid;
            continue;
        }

        for (int j=0; j<found; ++j) {
            const int idx = indxs[j];
            const float d2 = d2s[j];
            if (d2 < mind2[idx]) {
                mind2[idx] = d2;
                corresp[idx] = vid;
            }
            if (!covered[idx]) {
                covered[idx] = 1;
                num_covered -= 1;
            }
        }
    }
    
    // EXPORT CORRESP IF NEEDED DOWNSTREAM... (CD.corresp or something...)
}




/* HELPER FUNCTIONS */

float Rosa::similarity_metric(const Eigen::Vector3f& p1, const Eigen::Vector3f& v1, const Eigen::Vector3f& p2, const Eigen::Vector3f& v2, float range_r, float scale=5.0f) {
    const Eigen::Vector3f d = p1 - p2;
    const float dot12_raw = v1.dot(v2);
    const float dot12 = std::max(0.0f, std::min(1.0f, dot12_raw));
    if (dot12 == 0.0f) return 0.0f; // antiparallel or orthogonal

    const float proj1 = d.dot(v1);
    const float proj2 = -d.dot(v2);

    float dist1 = (d + scale * proj1 * v1).norm() / range_r;
    float dist2 = (d - scale * proj2 * v2).norm() / range_r;

    auto bump = [](float t) {
        if (t>1.0f) return 0.0f;
        const float t2 = t * t;
        const float t3 = t2 * t;
        return 2.0f * t3 - 3.0f * t2 + 1.0f;
    };

    const float k1 = bump(dist1);
    if (k1 == 0.0f) return 0.0f;
    const float k2 = bump(dist2);
    if (k2 == 0.0f) return 0.0f;

    const float align2 = dot12 * dot12;
    return std::min(k1,k2) * align2;
}

std::vector<int> Rosa::compute_active_samples(const int seed, const Eigen::Vector3f& p, const Eigen::Vector3f& n) {
    std::vector<char> state(CD.pcd_size_, 0); // 0=unknown, 1=queued, 2=visited
    std::vector<int> out;
    std::vector<int> stack;

    auto in_slab = [&](int i)->bool {
        const auto& q = CD.pts_->points[i];
        float d = n[0]*(p[0]-q.x) + n[1]*(p[1]-q.y) + n[2]*(p[2]-q.z);
        return std::abs(d) < leaf_size;
    };

    if (!in_slab(seed)) return out;

    state[seed] = 1;
    stack.push_back(seed);

    while(!stack.empty()) {
        int cur = stack.back();
        stack.pop_back();
        state[cur] = 2;
        out.push_back(cur);

        for (int nb : CD.simi_nbs[cur]) {
            if (state[nb] == 0 && in_slab(nb)) {
                state[nb] = 1;
                stack.push_back(nb);
            }
        }
    }
    return out;
}

Eigen::Vector3f Rosa::compute_symmetrynormal(const std::vector<int>& idxs) {
    const int m = (int)idxs.size();
    if (m==0) return Eigen::Vector3f::UnitX();

    Eigen::Vector3f sum_v = Eigen::Vector3f::Zero();
    Eigen::Matrix3f sum_outer = Eigen::Matrix3f::Zero();

    for (int idx : idxs) {
        const auto v = CD.nrms_->points[idx].getNormalVector3fMap();
        float s2 = v.squaredNorm();
        const float inv_len = (s2 > 1e-20) ? 1.0f / std::sqrt(s2) : 0.0f;
        const Eigen::Vector3f vn = inv_len * v;
        sum_v += vn;
        sum_outer.noalias() += vn * vn.transpose();
    }

    const float inv_m = 1.0f / static_cast<float>(m);
    const Eigen::Vector3f mu = sum_v * inv_m;
    Eigen::Matrix3f M = sum_outer * inv_m - (mu * mu.transpose()); //covariance
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(M);
    return es.eigenvectors().col(0);
}   

float Rosa::symmnormal_variance(Eigen::Vector3f& symm_nrm, std::vector<int>& idxs) {
    const int m = (int)idxs.size();
    if (m==0) return 0.0f;

    float sum_a = 0.0f;
    float sum_a2 = 0.0f;

    for (int idx : idxs) {
        const auto v = CD.nrms_->points[idx].getNormalVector3fMap();
        const float s2 = v.squaredNorm();
        const float inv_len = (s2 > 1e-20f) ? 1.0f / std::sqrt(s2) : 0.0f;
        
        const float a = v.dot(symm_nrm) * inv_len;
        sum_a += a;
        sum_a2 += a * a;
    }

    const float inv_m = 1.0f / static_cast<float>(m);
    const float mu = sum_a * inv_m;
    float var = sum_a2 * inv_m - mu * mu;
    if (m > 1) {
        var *= static_cast<float>(m) / static_cast<float>(m-1);
    }
    return var;
}

Eigen::Vector3f Rosa::cov_eigs_from_normals(const std::vector<int>& idxs) {
    Eigen::Vector3f sum_v = Eigen::Vector3f::Zero();
    Eigen::Matrix3f sum_outer = Eigen::Matrix3f::Zero();

    for (int i : idxs) {
        const auto v = CD.nrms_->points[i].getNormalVector3fMap();
        const float s2 = v.squaredNorm();
        const float inv_len = (s2 > 1e-20) ? 1.0f / std::sqrt(s2) : 0.0f;
        const Eigen::Vector3f vn = v * inv_len;
        sum_v += vn;
        sum_outer.noalias() += vn * vn.transpose();
    }

    const float inv_m = 1.0f / std::max<size_t>(1, idxs.size());
    const Eigen::Vector3f mu = sum_v * inv_m;
    const Eigen::Matrix3f C = sum_outer * inv_m - mu * mu.transpose();
    
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(C);
    return es.eigenvalues(); // returns eigenvalues in ascending order...
}

Eigen::Vector3f Rosa::closest_projection_point(const std::vector<int>& idxs) {
    if (idxs.empty()) return Eigen::Vector3f::Constant(1e8);

    Eigen::Matrix3f M = Eigen::Matrix3f::Zero();
    Eigen::Vector3f B = Eigen::Vector3f::Zero();

    for (int i : idxs) {
        const auto p = CD.pts_->points[i].getVector3fMap();
        const auto v = CD.nrms_->points[i].getNormalVector3fMap();
        const float s2 = v.squaredNorm();
        const float inv_len = (s2 > 1e-20f) ? (1.0 / std::sqrt(s2)) : 0.0f;
        const Eigen::Vector3f vn = v * inv_len;
        
        const Eigen::Matrix3f P = Eigen::Matrix3f::Identity() - (vn * vn.transpose());
        M.noalias() += P;
        B.noalias() += P * p;
    }

    const float tr = M.trace();
    M.diagonal().array() += 1e-6f * std::max(tr, 1e-6f);

    Eigen::Vector3f X;
    Eigen::LLT<Eigen::Matrix3f> llt(M);
    if (llt.info() == Eigen::Success) {
        X = llt.solve(B);
    }
    else {
        Eigen::LDLT<Eigen::Matrix3f> ldlt(M);
        if (ldlt.info() == Eigen::Success) {
            X = ldlt.solve(B);
        }
        else {
            X = Eigen::Vector3f::Constant(1e8f);
        }
    }
    return X;
}

bool Rosa::local_line_fit(const std::vector<int>& nb, Eigen::Vector3f& mean_out, Eigen::Vector3f& dir_out, float& conf_out, const int min_nb) {
    const int m = static_cast<int>(nb.size());
    const float EPS = 1e-6f;
    if (m < min_nb) {
        mean_out.setZero();
        dir_out = Eigen::Vector3f::UnitX();
        conf_out = 0.0f;
        return false;
    }

    Eigen::Vector3f mean = Eigen::Vector3f::Zero();
    for (int j : nb) {
        mean += pset.row(j).transpose();
    }
    mean /= static_cast<float>(m);
    
    Eigen::Matrix3f C = Eigen::Matrix3f::Zero();
    for (int j : nb) {
        Eigen::Vector3f q = pset.row(j).transpose() - mean;
        C.noalias() += q * q.transpose(); 
    }
    C /= static_cast<float>(m);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(C);
    if (es.info() != Eigen::Success) {
        mean_out = mean;
        dir_out = Eigen::Vector3f::UnitX();
        conf_out = 0.0f;
        return false;
    }

    const Eigen::Vector3f eval = es.eigenvalues();
    const Eigen::Matrix3f evec = es.eigenvectors();

    const float denom = std::max(EPS, eval.sum());
    mean_out = mean;
    dir_out = evec.col(2); // principal direction (largest eigenvalue)
    const float nrm = dir_out.norm();

    if (nrm > EPS) {
        dir_out /= nrm;
    }
    else {
        dir_out = Eigen::Vector3f::UnitX();
    }

    conf_out = eval(2) / denom;
    return true;
} 



