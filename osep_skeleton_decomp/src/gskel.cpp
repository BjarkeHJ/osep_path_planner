/* 

Main algorithm for global incremental skeletonization
a topologically preserving representation of the structure

Note: If I want to utilize the KF filter after "frozen" is triggered 
I must change such that when the position (of Vertex) is changed it changes the KF state as well...

TODO: Downsample the skeleton vertices further to extract more meaningfull vertices and longer edges (smaller vertex set)
      Long branches have few vertices: linear fitting ish...

*/

#include <gskel.hpp>

GSkel::GSkel(const GSkelConfig& cfg) : cfg_(cfg) {
    /* Constructor - Init data structures etc... */
    GD.new_cands.reset(new pcl::PointCloud<pcl::PointXYZ>);
    GD.global_vers_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    GD.global_vers_cloud->points.reserve(10000);
    
    running = 1;
}

bool GSkel::gskel_run() {
    auto ts = std::chrono::high_resolution_clock::now();

    RUN_STEP(increment_skeleton);
    RUN_STEP(graph_adj);
    RUN_STEP(mst);
    RUN_STEP(vertex_merge);
    RUN_STEP(prune);
    RUN_STEP(smooth_vertex_positions);
    RUN_STEP(update_branches);

    std::cout << "Number of branches: " << GD.branches.size() << std::endl;
    for (int i=0; i<(int)GD.branches.size(); ++i) {
        std::cout << "Branch " << i << " - Number of vertices: " << GD.branches[i].size() << std::endl; 
    }

    RUN_STEP(vid_manager);

    auto te = std::chrono::high_resolution_clock::now();
    auto telaps = std::chrono::duration_cast<std::chrono::milliseconds>(te-ts).count();
    std::cout << "[GSKEL] Time Elapsed: " << telaps << " ms" << std::endl;
    return running;
}

bool GSkel::increment_skeleton() {
    if (!GD.new_cands || GD.new_cands->points.empty()) return 0;

    // Reset update flags in global skeleton
    for (auto& vg : GD.global_vers) {
        vg.pos_update = false;
        vg.type_update = false;
    }

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
            gver.obs_count++;

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
            new_ver.obs_count++;
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
    for (auto& v : GD.prelim_vers) {
        if (!pcl::isFinite(v.position)) continue;

        if (v.just_approved && v.position.getVector3fMap().z() > cfg_.gnd_th) {
            GD.global_vers.emplace_back(v); // copy to global vertices
            GD.new_vers_indxs.push_back(GD.global_vers.size() - 1);
            v.just_approved = false;
        }
    }

    GD.gskel_size = GD.global_vers.size();

    if (GD.new_vers_indxs.size() == 0) {
        // std::cout << "No new vertices!" << std::endl;
        return 0; // No reason to proceed with the pipeline 
    }
    else return 1;
}

bool GSkel::graph_adj() {
    if (!GD.global_vers_cloud) return 0;

    build_cloud_from_vertices();
    std::vector<std::vector<int>> new_adj(GD.global_vers_cloud->points.size());
    kd_tree_->setInputCloud(GD.global_vers_cloud);
    const int K = 10;
    // const float max_dist_th = 2.5f * cfg_.fuse_dist_th;
    const float max_dist_th = 3.0f * cfg_.fuse_dist_th;

    std::vector<int> indices;
    std::vector<float> dist2;
    indices.reserve(K);
    dist2.reserve(K);
    for (size_t i=0; i<GD.gskel_size; ++i) {
        indices.clear();
        dist2.clear();
        const auto& vq = GD.global_vers[i];
        int n_nbs = kd_tree_->nearestKSearch(vq.position, K, indices, dist2);
        for (int j=1; j<n_nbs; ++j) {
            int nb_idx = indices[j];
            const auto& vnb = GD.global_vers[nb_idx];
            float dist_to_nb = (vq.position.getVector3fMap() - vnb.position.getVector3fMap()).norm();

            if (dist_to_nb > max_dist_th) continue;

            bool is_good_nb = true;
            for (int k=1; k<n_nbs; ++k) {
                if (k==j) continue;

                int other_nb_idx = indices[k];
                const auto& vnb_2 = GD.global_vers[other_nb_idx];
                float dist_nb_to_other = (vnb.position.getVector3fMap() - vnb_2.position.getVector3fMap()).norm();
                float dist_to_other = (vq.position.getVector3fMap() - vnb_2.position.getVector3fMap()).norm();

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
            const Eigen::Vector3f& ver_i = GD.global_vers[i].position.getVector3fMap();
            const Eigen::Vector3f& ver_nb = GD.global_vers[nb].position.getVector3fMap();
            const float weight = (ver_i - ver_nb).norm();
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
    // graph_decomp();
    return size_assert();
}

bool GSkel::vertex_merge() {
    int N_new = GD.new_vers_indxs.size();
    if (GD.gskel_size == 0 || N_new == 0) return 0;
    if (static_cast<float>(N_new) / static_cast<float>(GD.gskel_size) > 0.5) return 1; // dont prune in beginning...

    // auto is_joint = [&](int idx) {
    //     return std::find(GD.joints.begin(), GD.joints.end(), idx) != GD.joints.end();
    // };

    std::set<int> to_delete;

    for (int new_id : GD.new_vers_indxs) {
        if (new_id < 0 || new_id >= (int)GD.gskel_size) continue;
        if (to_delete.count(new_id)) continue;

        const auto& nbrs = GD.global_adj[new_id];
        for (int nb_id : nbrs) {
            if (nb_id < 0 || nb_id >= (int)GD.gskel_size) continue;
            if (new_id == nb_id || to_delete.count(nb_id)) continue;

            bool do_merge = false;
            // if (is_joint(new_id) && is_joint(nb_id)) {
            if (GD.global_adj[new_id].size() > 2 && GD.global_adj[nb_id].size() > 2) {
                do_merge = true;
                // GD.joints.erase(std::remove(GD.joints.begin(), GD.joints.end(), nb_id), GD.joints.end());
            }

            const auto &Vi = GD.global_vers[new_id];
            const auto &Vj = GD.global_vers[nb_id];
            float dist = (Vi.position.getVector3fMap() - Vj.position.getVector3fMap()).norm();

            if (!do_merge && dist < 0.5f * cfg_.fuse_dist_th) {
                do_merge = true;
            }

            if (!do_merge) continue;

            merge_into(nb_id, new_id); // Keeps existing and deletes new_id (after merge)

            to_delete.insert(new_id);
            break;
        }
    }

    if (to_delete.empty()) return 1; // end with success - no need to merge

    for (auto it = to_delete.rbegin(); it != to_delete.crend(); ++it) {
        const int del = *it;
        if (del < 0 || del >= static_cast<int>(GD.global_vers.size())) continue;
        
        GD.global_vers.erase(GD.global_vers.begin() + del);
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
    // graph_decomp();
    return size_assert();
}

bool GSkel::prune() {
    int N_new = GD.new_vers_indxs.size();
    if (GD.gskel_size == 0 || N_new == 0) return 0;
    if (static_cast<float>(N_new) / static_cast<float>(GD.gskel_size) > 0.5) return 1; // dont prune in beginning...

    // std::set<int> joint_set(GD.joints.begin(), GD.joints.end());
    // std::set<int> new_set(GD.new_vers_indxs.begin(), GD.new_vers_indxs.end());
    std::vector<int> to_delete;

    for (int v : GD.new_vers_indxs) {
        if (v < 0 || v >= static_cast<int>(GD.global_adj.size())) continue;
        if (GD.global_adj[v].size() != 1) continue; // not a leaf
        int nb = GD.global_adj[v][0];
        if (GD.global_adj[nb].size() >= 3) {
            to_delete.push_back(v);
        }
    }

    if (to_delete.empty()) return 1; // exit w. success

    // for (int leaf : GD.leafs) {
    //     if (!new_set.count(leaf)) continue; // not a new leaf
    //     if (GD.global_adj[leaf].size() != 1) continue; // sanity check...

    //     int nb = GD.global_adj[leaf][0];
    //     if (joint_set.count(nb)) {
    //         to_delete.push_back(leaf);
    //     }
    // }

    std::sort(to_delete.rbegin(), to_delete.rend());
    for (int idx : to_delete) {
        GD.global_vers.erase(GD.global_vers.begin() + idx);
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
    // graph_decomp();
    return 1;
}

bool GSkel::smooth_vertex_positions() {
    if (GD.gskel_size == 0 || GD.global_adj.size() == 0) return 0;

    std::vector<pcl::PointXYZ> new_pos(GD.gskel_size);

    for (size_t i=0; i<GD.gskel_size; ++i) {
        auto &v = GD.global_vers[i];
        auto &nbrs = GD.global_adj[i];

        new_pos[i] = v.position;
        if (v.type == 1 || v.type == 3) continue;
        if (v.smooth_iters <= 0) continue;

        Eigen::Vector3f sum = Eigen::Vector3f::Zero();
        int cnt = 0;
        for (int j : nbrs) {
            if (j < 0 || j >= (int)GD.gskel_size) continue;
            const auto& p = GD.global_vers[j].position;
            sum += p.getVector3fMap();
            cnt++;
        }
        if (cnt == 0) continue; // nothing to average - no neighbors

        Eigen::Vector3f avg = sum / static_cast<float>(nbrs.size());
        new_pos[i].getVector3fMap() = (1.0f - cfg_.vertex_smooth_coef)*v.position.getVector3fMap() + cfg_.vertex_smooth_coef*avg;
        v.pos_update = true;
        --v.smooth_iters;
    }

    for (size_t i=0; i<GD.gskel_size; ++i) {
        GD.global_vers[i].position = new_pos[i];
    }

    return 1;
}

bool GSkel::update_branches() {
    if (GD.gskel_size == 0) return 0;
    graph_decomp(); // assign vertex types (GD:joints, GD.leafs)
    
    auto& V = GD.global_vers;
    auto& adj = GD.global_adj;
    auto& BR = GD.branches;

    // Drop empty branches...
   {
    std::vector<std::vector<int>> compact;
    compact.reserve(BR.size());
    for (auto &b : BR) {
        if (!b.empty()) compact.push_back(std::move(b)); // move OK
    }
    BR.swap(compact); // <-- ALWAYS swap

    // rebuild bid/bpos since indices may have changed
    for (auto& vx : GD.global_vers) { vx.bid = -1; vx.bpos = -1; }
    for (int b = 0; b < (int)BR.size(); ++b)
        for (int i = 0; i < (int)BR[b].size(); ++i) {
            int v = BR[b][i];
            GD.global_vers[v].bid  = b;
            GD.global_vers[v].bpos = i;
        }
    }

    // Attach new vertices
    for (int v : GD.new_vers_indxs) {
        if (v < 0 || v >= (int)V.size()) continue;
        if (V[v].bid != -1) continue; // already assigned somehow...

        int nb_in_branch = 0;
        int bid_hit = -1;
        int pos_left = -1;
        int pos_right = -1;

        // Count adjacent nbs that already have a branch 
        for (int nb : adj[v]) {
            if (nb < 0 || nb >= (int)V.size()) continue;
            if (V[nb].bid != -1) {
                nb_in_branch++;
                bid_hit = V[nb].bid;
                if (pos_left == -1) {
                    pos_left = V[nb].bpos;
                }
                else {
                    pos_right = V[nb].bpos;
                }
            } 
        }

        if (nb_in_branch == 0) continue; // not enough info
        if (bid_hit < 0 || bid_hit >= (int)BR.size()) continue;

        auto& br = BR[bid_hit];
        if (br.empty()) continue;

        if (nb_in_branch == 1) {
            // likely an extension to an endpoint (new leaf)
            int front = br.front();
            // int back = br.back();
            bool touches_front = std::find(adj[front].begin(), adj[front].end(), v) != adj[front].end();
            int side = touches_front ? -1 : +1;
            extend_branch(bid_hit, side, std::vector<int>{v});
        }
        else {
            // two nbs in the same branch -> v lies between them -> splice between indices
            if (pos_left < 0 || pos_right < 0) continue;
            int i0 = std::min(pos_left, pos_right);
            br.insert(br.begin() + (i0+1), v);
            for (int i=i0; i<(int)br.size(); ++i) {
                int vv = br[i];
                V[vv].bid = bid_hit;
                V[vv].bpos = i;
            }
        }
    }

    // Start/repair branches from unassigned endpoints
    std::vector<int> endpts;
    endpts.reserve(GD.joints.size() + GD.leafs.size());
    endpts.insert(endpts.end(), GD.leafs.begin(), GD.leafs.end());
    endpts.insert(endpts.end(), GD.joints.begin(), GD.joints.end());
    std::sort(endpts.begin(), endpts.end());
    // endpts.erase(std::unique(endpts.begin(), endpts.end()), endpts.end());
    
    for (int s : endpts) {
        if (s < 0 || s >= (int)V.size()) continue;
        if (V[s].bid != -1) continue;

        int first_nb = -1;
        for (int nb : adj[s]) {

            first_nb = nb;
            if (V[nb].bid == -1) break;
        }
        if (first_nb < 0) continue;

        std::vector<int> path;
        path.push_back(s);

        int prev = s;
        int curr = first_nb;
        int hit_bid = -1;
        while (true) {
            path.push_back(curr);
            if (V[curr].bid != -1) {
                hit_bid = V[curr].bid;
                break; // hit existing branch
            }

            if (is_endpoint(V[curr]) && curr != s) {
                break; // hit another endpoint
            }

            int nxt = -1;
            for (int nn : adj[curr]) {
                if (nn != prev) {
                    nxt = nn;
                    break; // found new in chain
                }
            }
        
            if (nxt < 0) break; // unable to find new in chain... -> break loop

            prev = curr;
            curr = nxt;
        }

        if (path.size() < 2) continue;

        if (hit_bid == -1) {
            // ended on a fresh endpoint -> new branch!
            (void)add_branch(path);
        }
        else {
            // hit existing branch at 'curr'
            auto& br = BR[hit_bid];
            if (br.empty()) continue;
            int front = br.front();
            int back = br.back();
            int side = (+1); // default to back
            if (curr == front) side = -1; // hit the front
            else if (curr == back) side = +1; // hit the back
            else {
                // hit interior of branch -> split and insert in the left part
                int ip = V[curr].bpos;
                std::vector<int> left(br.begin(), br.begin()+ip+1);
                std::vector<int> right(br.begin()+ip, br.end());

                br = std::move(left);
                for (int i=0; i<(int)br.size(); ++i) {
                    int vv = br[i];
                    V[vv].bid = hit_bid;
                    V[vv].bpos = i;
                }

                if (right.size() >= 2) {
                    // int bid2 = add_branch(right);
                    (void)add_branch(right); // add branch from the right part
                }

                side = +1;
            }

            if (!path.empty() && path.back() == curr) {
                path.pop_back();
            }

            extend_branch(hit_bid, side, path);
        }
    }

    return 1;
}

bool GSkel::vid_manager() {
    for (int idx : GD.new_vers_indxs) {
        if (idx < 0 || idx >= (int)GD.global_vers.size()) continue;
        auto &v = GD.global_vers[idx];
        if (v.vid < 0) {
            v.vid = GD.next_vid++;
        }
    }

    for (size_t idx=0; idx<GD.gskel_size; ++idx) {
        auto& v = GD.global_vers[idx];
        const auto& nbrs = GD.global_adj[idx];
        v.nb_ids.clear();
        for (int nb : nbrs) {
            v.nb_ids.push_back(nb);
        }
    }

    build_cloud_from_vertices(); // To publish correct cloud 
    return 1;
}

/* Helpers */
void GSkel::build_cloud_from_vertices() {
    if (GD.global_vers.empty()) return;
    GD.global_vers_cloud->clear();
    for (const auto& v : GD.global_vers) {
        if (!pcl::isFinite(v.position)) {
            std::cout << "INVALID POSITION!!" << std::endl;
        }
        GD.global_vers_cloud->points.push_back(v.position);
    }

    GD.global_vers_cloud->width  = static_cast<uint32_t>(GD.global_vers_cloud->points.size());
    GD.global_vers_cloud->height = 1;
    GD.global_vers_cloud->is_dense = true;

    if (GD.global_vers.size() != GD.global_vers_cloud->points.size()) {
        std::cout << "NOT SAME SIZE???" << std::endl;
    }
}

void GSkel::graph_decomp() {
    GD.joints.clear();
    GD.leafs.clear();

    const int N = GD.global_vers.size();
    for (int i=0; i<N; ++i) {
        auto &vg = GD.global_vers[i];
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
            break;
        default:
            if (degree > 2) {
                GD.joints.push_back(i);
                v_type = 3;
            }
            break;
        }

        // update type
        if (vg.type != v_type) vg.type_update = true;
        vg.type = v_type;
    }
}

void GSkel::merge_into(int keep, int del) {
    auto& Vi = GD.global_vers[keep];
    auto& Vj = GD.global_vers[del];

    int tot = Vi.obs_count + Vj.obs_count;
    if (tot == 0) tot = 1;
    Vi.position.getVector3fMap() = (Vi.position.getVector3fMap() * Vi.obs_count + Vj.position.getVector3fMap() * Vj.obs_count) / static_cast<float>(tot);
    Vi.obs_count = tot;
    Vi.pos_update = true; // position updated

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
    const int B = static_cast<int>(GD.global_adj.size());
    const int C = static_cast<int>(GD.gskel_size);
    const bool ok = (A == B) && (B == C);
    return ok;
}

bool GSkel::is_endpoint(const Vertex& v) {
    return (v.type == 1 || v.type == 3);
}

int GSkel::add_branch(const std::vector<int>& chain) {
    if (chain.size() < 2) return -1; // replace with max_branch_size???
    int bid = (int)GD.branches.size();
    GD.branches.push_back(chain);
    for (int i=0; i<(int)chain.size(); ++i) {
        int v = chain[i];
        GD.global_vers[v].bid = bid; // set vertex branch id
        GD.global_vers[v].bpos = i; // ser branch position id
    }
    return bid;
}

void GSkel::extend_branch(int bid, int side, const std::vector<int>& extra) {
    // Insert a small sequence at a branch end (front/back)
    // side: -1 = front, +1 = back
    if (extra.empty()) return;
    auto& br = GD.branches[bid];

    if (side < 0) {
        // insert at front, keep order
        br.insert(br.begin(), extra.begin(), extra.end());
        // reindex
        for (int i=0; i<(int)br.size(); ++i) {
            int v = br[i];
            GD.global_vers[v].bid = bid;
            GD.global_vers[v].bpos = i;
        }
    }
    else {
        // append at back
        int start = (int)br.size();
        br.insert(br.end(), extra.begin(), extra.end());
        for (int i=start; i<(int)br.size(); ++i) {
            int v = br[i];
            GD.global_vers[v].bid = bid;
            GD.global_vers[v].bpos = i;
        }
    }
}

void GSkel::merge_branch(int bidA, int A_end, std::vector<int> mid, int bidB, int B_end) {
    // If two branches are connected by 'mid' sequence, merge them into A
    // A_end = -1 front, +1 back
    // B_end = -1 front, +1 back
    if (bidA == bidB) return;
    auto& A = GD.branches[bidA];
    auto& B = GD.branches[bidB];
    if (A.empty() || B.empty()) return;

    if (A_end < 0) {
        A.insert(A.begin(), mid.begin(), mid.end());
        if (B_end < 0) std::reverse(B.begin(), B.end());
        A.insert(A.end(), B.begin(), B.end());
    }
    else {
        if (B_end > 0) std::reverse(B.begin(), B.end());
        A.insert(A.end(), mid.begin(), mid.end());
        A.insert(A.end(), B.begin(), B.end());
    }

    // reindex A
    for (int i=0; i<(int)A.size(); ++i) {
        int v = A[i];
        GD.global_vers[v].bid = bidA;
        GD.global_vers[v].bpos = i;
    }

    // invalidate B
    B.clear();
    for (auto& vv : GD.global_vers) {
        if (vv.bid == bidB) {
            vv.bid = -1;
            vv.bpos = -1;
        }
    }

}
