/* 

Main algorithm for global incremental skeletonization
a topologically preserving representation of the structure

Note: If I want to utilize the KF filter after "frozen" is triggered 
I must change such that when the position (of Vertex) is changed it changes the KF state as well...

*/

#include <gskel.hpp>

GSkel::GSkel(const GSkelConfig& cfg) : cfg_(cfg) {
    /* Constructor - Init data structures etc... */
    GD.new_cands.reset(new pcl::PointCloud<pcl::PointXYZ>);
    GD.global_vers_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    
    running = 1;
}

bool GSkel::gskel_run() {
    // auto ts = std::chrono::high_resolution_clock::now();

    RUN_STEP(increment_skeleton);
    RUN_STEP(graph_adj);
    RUN_STEP(mst);
    RUN_STEP(vertex_merge);
    RUN_STEP(prune);
    RUN_STEP(smooth_vertex_positions);
    RUN_STEP(extract_branches);

    // auto te = std::chrono::high_resolution_clock::now();
    // auto telaps = std::chrono::duration_cast<std::chrono::milliseconds>(te-ts).count();
    // std::cout << "[GSKEL] Time Elapsed: " << telaps << " ms" << std::endl;
    return running;
}

bool GSkel::increment_skeleton() {
    if (!GD.new_cands || GD.new_cands->points.empty()) return 0;

    // Reset flags
    for (auto& v : GD.prelim_vers) {
        v.just_approved = false;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr prelim_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    prelim_cloud->points.reserve(GD.prelim_vers.size());
    for (const auto& v : GD.prelim_vers) {
        prelim_cloud->points.push_back(v.position);
    }

    if (!prelim_cloud->points.empty()) {
        kd_tree_->setInputCloud(prelim_cloud);
    }

    const float fuse_r2 = cfg_.fuse_dist_th * cfg_.fuse_dist_th;

    std::vector<int> idx_buff;
    std::vector<float> dist2_buff;
    idx_buff.reserve(16);
    dist2_buff.reserve(16);
    std::vector<pcl::PointXYZ> spawned_this_tick;
    spawned_this_tick.reserve(128);
    for (auto& pt : GD.new_cands->points) {
        Eigen::Vector3f ver = pt.getVector3fMap();
        if (!ver.allFinite()) continue;

        int chosen_idx = -1;
        float chosen_dist2 = std::numeric_limits<float>::max();
        bool any_in_radius = false;
        bool frozen_in_radius = false;
        if (!prelim_cloud->points.empty()) {
            idx_buff.clear();
            dist2_buff.clear();
            if (kd_tree_->radiusSearch(pt, cfg_.fuse_dist_th, idx_buff, dist2_buff) > 0) {
                any_in_radius = true;
                // pick nearest "non-frozen" candidate
                for (size_t k=0; k<idx_buff.size(); ++k) {
                    const int i = idx_buff[k];
                    auto& gver = GD.prelim_vers[i];
                    if (gver.frozen) {
                        frozen_in_radius = true;
                        continue;
                    }
                    if (dist2_buff[k] < chosen_dist2) {
                        chosen_dist2 = dist2_buff[k];
                        chosen_idx = i;
                    }
                }
            }
        }

        // To prevent spawning multiple at the same place this tick
        auto near_spawned = [&]() {
            for (const auto&s : spawned_this_tick) {
                Eigen::Vector3f sv = const_cast<pcl::PointXYZ&>(s).getVector3fMap();
                if ((sv - ver).squaredNorm() <= fuse_r2) return true;
            }
            return false;
        };

        if (chosen_idx >= 0 && chosen_dist2 <= fuse_r2) {
            auto& gver = GD.prelim_vers[chosen_idx];
            gver.kf.update(ver, Q, R);
            if (!gver.kf.x.allFinite()) {
                gver.marked_for_deletion = true;
                continue;
            }
            gver.position.getVector3fMap() = gver.kf.x; // overwrite position
            gver.obs_coubt++;

            const float trace = gver.kf.P.trace();
            if (!gver.conf_check && trace < cfg_.fuse_conf_th) {
                gver.conf_check = true;
                gver.just_approved = true;
                gver.frozen = true;
                gver.unconf_check = 0;
            }
            else {
                gver.unconf_check++;
            }

            if (gver.unconf_check > cfg_.max_obs_wo_conf) {
                gver.marked_for_deletion = true;
            }
        }
        else if (any_in_radius || frozen_in_radius) {
            // area occupied
            continue;
        }
        else if (!near_spawned()) {
            Vertex new_ver;
            Eigen::Matrix3f P0 = Eigen::Matrix3f::Identity();
            new_ver.kf.initFrom(ver, P0);
            new_ver.position.getVector3fMap() = ver;
            new_ver.obs_coubt++;
            new_ver.smooth_iters = cfg_.niter_smooth_vertex;
            GD.prelim_vers.emplace_back(std::move(new_ver));
            spawned_this_tick.push_back(pt);
        }
    }

    // Remove vertices marked for deletion
    GD.prelim_vers.erase(
        std::remove_if(
            GD.prelim_vers.begin(),
            GD.prelim_vers.end(),
            [](const Vertex& v) {
                return v.marked_for_deletion;
            }
        ),
        GD.prelim_vers.end()
    );

    // Pass confident vertices to the global skeleton
    GD.new_vers_indxs.clear();
    GD.global_vers.reserve(GD.global_vers.size() + GD.prelim_vers.size());
    GD.global_vers_cloud->points.reserve(GD.global_vers_cloud->points.size() + GD.prelim_vers.size());

    for (auto& v : GD.prelim_vers) {
        if (v.just_approved && v.position.getVector3fMap().z() > cfg_.gnd_th) {
            // GD.global_vers.emplace_back(std::move(v)); // move (remove from prelim)
            GD.global_vers.emplace_back(v); // copy
            GD.global_vers_cloud->points.emplace_back(v.position);
            GD.new_vers_indxs.push_back(GD.global_vers.size() - 1);
            v.just_approved = false;
            v.updated = true;
        }
    }

    {
        std::vector<Vertex> kept;
        kept.reserve(GD.global_vers.size());
    
        pcl::PointCloud<pcl::PointXYZ> clean;
        clean.points.reserve(GD.global_vers.size());
    
        for (auto& v : GD.global_vers) {
            const auto& p = v.position;
            if (!pcl::isFinite(p)) continue;  // drop NaN/Inf
            kept.emplace_back(std::move(v));
            clean.points.emplace_back(p);
        }
    
        GD.global_vers.swap(kept);
        *GD.global_vers_cloud = std::move(clean);
    
        GD.global_vers_cloud->is_dense = true;
        GD.global_vers_cloud->width  = static_cast<uint32_t>(GD.global_vers_cloud->points.size());
        GD.global_vers_cloud->height = 1;
    
        GD.gskel_size = GD.global_vers.size();
    }

    if (GD.new_vers_indxs.size() == 0) return 0;
    else return 1;
}

bool GSkel::graph_adj() {
    if (!GD.global_vers_cloud || GD.global_vers_cloud->empty()) return 0;

    std::vector<std::vector<int>> new_adj(GD.global_vers_cloud->points.size());
    kd_tree_->setInputCloud(GD.global_vers_cloud);
    const int K = 10;
    const float max_dist_th = 2.5f * cfg_.fuse_dist_th;

    std::vector<int> indices;
    std::vector<float> dist2;
    indices.reserve(K);
    dist2.reserve(K);
    for (size_t i=0; i<GD.gskel_size; ++i) {
        indices.clear();
        dist2.clear();
        int n_nbs = kd_tree_->nearestKSearch(GD.global_vers_cloud->points[i], K, indices, dist2);
        for (int j=1; j<n_nbs; ++j) {
            int nb_idx = indices[j];
            float dist_to_nb = (GD.global_vers[i].position.getVector3fMap() - GD.global_vers[nb_idx].position.getVector3fMap()).norm();

            if (dist_to_nb > max_dist_th) continue;

            bool is_good_nb = true;
            for (int k=1; k<n_nbs; ++k) {
                if (k==j) continue;

                int other_nb_idx = indices[k];
                float dist_nb_to_other = (GD.global_vers[nb_idx].position.getVector3fMap() - GD.global_vers[other_nb_idx].position.getVector3fMap()).norm();
                float dist_to_other = (GD.global_vers[i].position.getVector3fMap() - GD.global_vers[other_nb_idx].position.getVector3fMap()).norm();

                if (dist_nb_to_other < dist_to_nb && dist_to_other < dist_to_nb) {
                    is_good_nb = false;
                    break;
                }
            }

            if (is_good_nb) {
                new_adj[i].push_back(nb_idx);
                new_adj[nb_idx].push_back(i);
            }
        } 
    }

    GD.global_adj = new_adj; // replace old adjencency
    return size_assert();
}

bool GSkel::mst() {
    if (GD.gskel_size == 0 || GD.global_adj.empty()) return 0;

    std::vector<Edge> mst_edges;
    for (size_t i=0; i<GD.gskel_size; ++i) {
        for (int nb : GD.global_adj[i]) {
            if (nb <= (int)i) continue; // avoid bi-directional check
            Eigen::Vector3f ver_i = GD.global_vers[i].position.getVector3fMap();
            Eigen::Vector3f ver_nb = GD.global_vers[nb].position.getVector3fMap();
            float weight = (ver_i - ver_nb).norm();
            Edge new_edge;
            new_edge.u = i;
            new_edge.v = nb;
            new_edge.w = weight;
            mst_edges.push_back(new_edge);
        }
    }

    std::sort(mst_edges.begin(), mst_edges.end());
    UnionFind uf(GD.gskel_size);
    std::vector<std::vector<int>> mst_adj(GD.gskel_size);

    for (const auto& edge : mst_edges) {
        if (uf.unite(edge.u, edge.v)) {
            mst_adj[edge.u].push_back(edge.v);
            mst_adj[edge.v].push_back(edge.u);
        }
    }

    GD.global_adj = std::move(mst_adj);
    graph_decomp();
    // return 1;
    return size_assert();
}

bool GSkel::vertex_merge() {
    int N_new = GD.new_vers_indxs.size();
    if (GD.gskel_size == 0 || N_new == 0) return 0;
    if (static_cast<float>(N_new) / static_cast<float>(GD.gskel_size) > 0.5) return 0; // initialization of structure (change??)

    auto is_joint = [&](int idx) {
        return std::find(GD.joints.begin(), GD.joints.end(), idx) != GD.joints.end();
    };

    std::set<int> to_delete;

    for (int new_id : GD.new_vers_indxs) {
        if (new_id < 0 || new_id >= (int)GD.gskel_size) continue;
        if (to_delete.count(new_id)) continue;

        const auto& nbrs = GD.global_adj[new_id];
        for (int nb_id : nbrs) {
            if (nb_id < 0 || nb_id >= (int)GD.gskel_size) continue;
            if (new_id == nb_id || to_delete.count(nb_id)) continue;

            bool do_merge = false;
            if (is_joint(new_id) && is_joint(nb_id)) {
                do_merge = true;
                GD.joints.erase(std::remove(GD.joints.begin(), GD.joints.end(), nb_id), GD.joints.end());
            }

            const auto &Vi = GD.global_vers[new_id];
            const auto &Vj = GD.global_vers[nb_id];
            float dist = (Vi.position.getVector3fMap() - Vj.position.getVector3fMap()).norm();

            if (!do_merge && dist < 0.5f * cfg_.fuse_dist_th) {
                do_merge = true;
            }

            if (!do_merge) continue;

            merge_into(nb_id, new_id);
            to_delete.insert(new_id);
            break;
        }
    }

    if (to_delete.empty()) return 1; // end with success - no need to merge

    for (auto it = to_delete.rbegin(); it != to_delete.crend(); ++it) {
        const int del = *it;
        if (del < 0 || del > static_cast<int>(GD.global_vers.size())) continue;
        
        GD.global_vers.erase(GD.global_vers.begin() + del);
        GD.global_vers_cloud->points.erase(GD.global_vers_cloud->points.begin() + del);
        GD.global_adj.erase(GD.global_adj.begin() + del);

        for (auto &nbrs : GD.global_adj) {
            nbrs.erase(std::remove(nbrs.begin(), nbrs.end(), del), nbrs.end());
            for (auto &v : nbrs) {
                if (v > del) --v;
            }
            std::sort(nbrs.begin(), nbrs.end());
            nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
        }

        GD.new_vers_indxs.erase(std::remove(GD.new_vers_indxs.begin(), GD.new_vers_indxs.end(), del), GD.new_vers_indxs.end());
        for (auto &id : GD.new_vers_indxs) {
            if (id > del) --id;
        }
    }

    GD.gskel_size = GD.global_vers.size();
    graph_decomp();
    return size_assert();
}

bool GSkel::prune() {
    int N_new = GD.new_vers_indxs.size();
    if (GD.gskel_size == 0 || N_new == 0) return 0;
    if (static_cast<float>(N_new) / static_cast<float>(GD.gskel_size) > 0.5) return 0; // initialization of structure (change??)

    std::set<int> joint_set(GD.joints.begin(), GD.joints.end());
    std::set<int> new_set(GD.new_vers_indxs.begin(), GD.new_vers_indxs.end());
    std::vector<int> to_delete;

    for (int leaf : GD.leafs) {
        if (!new_set.count(leaf)) continue; // not a new leaf
        if (GD.global_adj[leaf].size() != 1) continue; // sanity check...

        int nb = GD.global_adj[leaf][0];
        if (joint_set.count(nb)) {
            to_delete.push_back(leaf);
        }
    }

    std::sort(to_delete.rbegin(), to_delete.rend());
    for (int idx : to_delete) {
        GD.global_vers.erase(GD.global_vers.begin() + idx);
        GD.global_vers_cloud->points.erase(GD.global_vers_cloud->points.begin() + idx);
        GD.global_adj.erase(GD.global_adj.begin() + idx);
        for (auto &nbrs : GD.global_adj) {
            nbrs.erase(std::remove(nbrs.begin(), nbrs.end(), idx), nbrs.end());
            for (auto &v : nbrs) {
                if (v > idx) --v;
            }
        }

        GD.new_vers_indxs.erase(std::remove(GD.new_vers_indxs.begin(), GD.new_vers_indxs.end(), idx), GD.new_vers_indxs.end());
        for (auto &id : GD.new_vers_indxs) {
            if (id > idx) -- id;
        }
    }
    GD.gskel_size = GD.global_vers.size();
    graph_decomp();
    return 1;
}

bool GSkel::smooth_vertex_positions() {
    if (GD.gskel_size == 0 || GD.global_adj.size() == 0) return 0;

    std::vector<pcl::PointXYZ> new_pos(GD.gskel_size);

    for (size_t i=0; i<GD.gskel_size; ++i) {
        auto &v = GD.global_vers[i];
        auto &nbrs = GD.global_adj[i];

        if (v.type == 1 || v.type == 3) {
            new_pos[i] = v.position;
        }

        if (v.smooth_iters > 0) {
            Eigen::Vector3f avg = Eigen::Vector3f::Zero();
            for (int j : nbrs) {
                avg += GD.global_vers[j].position.getVector3fMap();
            }
            avg /= static_cast<float>(nbrs.size());
            new_pos[i].getVector3fMap() = (1.0f - cfg_.vertex_smooth_coef)*v.position.getVector3fMap() + cfg_.vertex_smooth_coef*avg;

            --v.smooth_iters;
        }
        else {
            new_pos[i] = v.position;
        }
    }

    for (size_t i=0; i<GD.gskel_size; ++i) {
        GD.global_vers[i].position = new_pos[i];
    }

    GD.global_vers_cloud->clear();
    for (const auto &v : GD.global_vers) {
        GD.global_vers_cloud->points.emplace_back(v.position);
    }

    return 1;
}

bool GSkel::extract_branches() {
    if (GD.gskel_size == 0) return 0;
    GD.branches.clear();

    std::set<std::pair<int,int>> visited_edges;

    auto is_endpoint = [&](int vidx) {
        return GD.global_vers[vidx].type == 1 || GD.global_vers[vidx].type == 3;
    };

    for (int endp : GD.leafs) {
        for (int nb : GD.global_adj[endp]) {
            if (endp > nb) std::swap(endp, nb); // interpret edge ab and ba the same
            if (visited_edges.count({endp, nb})) continue;

            std::vector<int> branch;
            branch.push_back(endp);

            int prev = endp;
            int curr = nb;

            while (true) {
                branch.push_back(curr);
                if (is_endpoint(curr) && curr != endp) break; // stop at a different endpoint
                int next = -1;

                for (int nn : GD.global_adj[curr]) {
                    if (nn != prev) {
                        next = nn; // choose one nb
                        break;
                    }
                }

                if (next == -1) break; // dead end
                prev = curr;
                curr = next;
            }

            for (size_t i=1; i<branch.size(); ++i) {
                std::pair<int,int> edge{branch[i-1], branch[i]};
                visited_edges.insert(edge);
            }
            GD.branches.push_back(branch);
        }
    }

    // Repeat for joints...
    for (int endp : GD.joints) {
        for (int nb : GD.global_adj[endp]) {
            if (endp > nb) std::swap(endp, nb);
            if (visited_edges.count({endp, nb}));

            std::vector<int> branch;
            branch.push_back(endp);

            int prev = endp;
            int curr = nb;

            while (true) {
                branch.push_back(curr);
                if (is_endpoint(curr) && curr != endp) break;

                int next = -1;
                for (int nn : GD.global_adj[curr]) {
                    if (nn != prev) {
                        next = nn;
                        break;
                    }
                }

                if (next == -1) break;
                prev = curr;
                curr = next;
            }

            for (size_t i=1; i<branch.size(); ++i) {
                std::pair<int,int> edge{branch[i-1], branch[i]};
                visited_edges.insert(edge);
            }
            GD.branches.push_back(branch);
        }
    }

    std::sort(GD.branches.begin(), GD.branches.end(),
        [](const std::vector<int> &a, const std::vector<int> &b) {
            return a.size() < b.size();
        }
    );

    const size_t min_bl = cfg_.min_branch_length;
    GD.branches.erase(
        std::remove_if(
            GD.branches.begin(), GD.branches.end(),
            [min_bl](const std::vector<int>& branch) {
                return branch.size() < min_bl;
            }
        ),
        GD.branches.end()
    );
    return 1;
}

/* Helpers */
void GSkel::graph_decomp() {
    GD.joints.clear();
    GD.leafs.clear();

    int N = GD.global_vers.size();
    for (int i=0; i<N; ++i) {
        int degree = GD.global_adj[i].size();
        int v_type = 0;

        switch (degree)
        {
        case 1:
            GD.leafs.push_back(i);
            v_type = 1;
            break;
        case 2:
            v_type = 2;
        default:
            if (degree > 2) {
                GD.joints.push_back(i);
                v_type = 3;
            }
            v_type = 0; // stay 0 (invalid)
            break;
        }

        GD.global_vers[i].type = v_type;
    }
}

void GSkel::merge_into(int keep, int del) {
    auto& Vi = GD.global_vers[keep];
    auto& Vj = GD.global_vers[del];

    int tot = Vi.obs_coubt + Vj.obs_coubt;
    Vi.position.getVector3fMap() = (Vi.position.getVector3fMap() * Vi.obs_coubt + Vj.position.getVector3fMap() * Vj.obs_coubt) / static_cast<float>(tot);
    Vi.obs_coubt = tot;

    // Remap neighbors 
    for (int nb : GD.global_adj[del]) {
        if (nb == keep) continue;
        auto& keep_nbs = GD.global_adj[keep];
        if (std::find(keep_nbs.begin(), keep_nbs.end(), nb) == keep_nbs.end()) {
            keep_nbs.push_back(nb);
        }

        // Remap neighbor's neighbors
        auto &nbs_nb = GD.global_adj[nb];
        std::replace(nbs_nb.begin(), nbs_nb.end(), del, keep);
    }
}

bool GSkel::size_assert() {
    const int A = static_cast<int>(GD.global_vers.size());
    const int B = static_cast<int>(GD.global_vers_cloud->points.size());
    const int C = static_cast<int>(GD.global_adj.size());
    const int D = static_cast<int>(GD.gskel_size);

    const bool ok = (A == B) && (B == C) && (C == D);
    return ok;
}

