/* 

Main algorithm for the OSEP viewpoint manager

TODO: Incorporate 2d_local_costmap into the viewpoint sampling process.
TODO: Extract branches from the global skeleton (Move function from gskel)
TODO: 

*/

#include "viewpoint_manager.hpp"

ViewpointManager::ViewpointManager(const ViewpointConfig& cfg) : cfg_(cfg) {

    VD.global_skel.reserve(1000);
    VD.global_vpts.reserve(5000);
    VD.updated_vertices.reserve(100);
    running = 1;
}


bool ViewpointManager::viewpoint_run() {
    VD.gskel_size = VD.global_skel.size();
    RUN_STEP(fetch_updated_vertices);
    RUN_STEP(branch_extract);
    RUN_STEP(viewpoint_sampling);

    // RUN_STEP(branch_reduction);

    return running;
}

bool ViewpointManager::fetch_updated_vertices() {
    if (VD.gskel_size == 0) return 0;
    VD.updated_vertices.clear();
    for (const auto& v : VD.global_skel) {
        if (v.pos_update || v.type_update) {
            VD.updated_vertices.push_back(v.vid);
        }
    }

    // Rebuild graph cache
    const int N = VD.gskel_size;
    vid2idx_.clear();
    vid2idx_.reserve(N*2);
    idx2vid_.clear();
    idx2vid_.assign(N, -1);
    
    for (int i=0; i<N; ++i) {
        vid2idx_[VD.global_skel[i].vid] = i;
        idx2vid_[i] = VD.global_skel[i].vid;
    }
    
    VD.global_adj.assign(N, {});
    degree_.assign(N,0);
    is_endpoint_.assign(N,0);

    for (int i=0; i<N; ++i) {
        const auto& nbs = VD.global_skel[i].nb_ids;
        auto& out = VD.global_adj[i];
        out.reserve(nbs.size());
        for (int nb_vid : nbs) {
            auto it = vid2idx_.find(nb_vid);
            if (it != vid2idx_.end()) {
                out.push_back(it->second);
            }
        }
        degree_[i] = static_cast<int>(out.size());
    }
    for (int i=0; i<N; ++i) {
        int d = degree_[i];
        is_endpoint_[i] = (d == 1 || d > 2) ? 1 : 0;
    }
    return 1;
}

bool ViewpointManager::branch_extract() {
    const int N = VD.gskel_size;
    if (N == 0) return 0;
    const int branch_min_vers_ = 5;

    std::vector<int> updated_idxs;
    updated_idxs.reserve(VD.updated_vertices.size());
    for (int vid : VD.updated_vertices) {
        auto it = vid2idx_.find(vid);
        if (it != vid2idx_.end()) {
            updated_idxs.push_back(it->second);
        }
    }

    // const bool full_rebuild = VD.branches.empty() || updated_idxs.empty();
    const bool full_rebuild = VD.branches.empty();
    // const bool full_rebuild = true;

    if (full_rebuild) {
        VD.branches.clear();
        std::vector<char> allowed(N,1);
        std::unordered_set<std::pair<int,int>, PairHash> visited;
        visited.reserve(std::max(64, 2*N));

        for (int i=0; i<N; ++i) {
            if (!is_endpoint_[i]) continue; // start at endpoint
            for (int nb_i : VD.global_adj[i]) {
                auto br = walk_branch(i, nb_i, allowed, visited);
                if (!br.empty()) {
                    const bool enough_vers = (static_cast<int>(br.size()) - 1) >= branch_min_vers_;
                    if (enough_vers) VD.branches.emplace_back(std::move(br));
                }
            }
        }
        return 1;
    }

    // Incremental build
    std::vector<char> in_region(N,0);
    std::vector<int> stack;
    stack.reserve(256);

    auto push_idx = [&](int i) {
        if (i < 0 || i >= N) return;
        if (!in_region[i]) {
            in_region[i] = 1; // mark index as in region with updated vertex
            stack.push_back(i);
        }
    };

    // Seed the region with updated vertices and their neighbors
    for (int ui : updated_idxs) {
        push_idx(ui);
        for (int nb_i : VD.global_adj[ui]) {
            push_idx(nb_i);
        }
    }

    // DFS flood-fill the region and stopping at endpoints
    while (!stack.empty()) {
        int i = stack.back();
        stack.pop_back();
        if (is_endpoint_[i]) continue; // stop expanding the region at an endpoint (reached depth in search - trace back)
        for (int nb_i : VD.global_adj[i]) {
            push_idx(nb_i);
        }
    }

    bool any_in = false;
    for (char c : in_region) {
        if (c) {
            any_in = true;
            break;
        }
    }

    if (!any_in) return 1; // nothing to do...

    auto edge_key= [](int u, int v) {
        if (u > v) std::swap(u,v);
        return std::make_pair(u,v);
    };
    
    // Build the region edges
    std::unordered_set<std::pair<int,int>, PairHash> region_edges;
    region_edges.reserve(8 * (int)updated_idxs.size() + 64);
    for (int i=0; i<N; ++i) {
        if (!in_region[i]) continue; // focus on the region (skip non-update areas)
        for (int j : VD.global_adj[i]) {
            if (!in_region[j]) continue;
            if (j <= i) continue; // avoid duplicate edges (undirected...)
            region_edges.insert(edge_key(i,j));
        }
    }

    auto branch_hits_region = [&](const std::vector<int>& br_vids) -> bool {
        if (br_vids.size() < 2) return false;
        for (size_t k=1; k<br_vids.size(); ++k) {
            auto ita = vid2idx_.find(br_vids[k-1]);
            auto itb = vid2idx_.find(br_vids[k]);
            if (ita == vid2idx_.end() || itb == vid2idx_.end()) continue;
            if (region_edges.count(edge_key(ita->second, itb->second))) {
                return true;
            }
        }
        return false;
    };

    // Remove any previously stored branches that crosses region edges
    std::vector<std::vector<int>> kept;
    kept.reserve(VD.branches.size());
    for (auto& br : VD.branches) {
        if (!branch_hits_region(br)) {
            kept.emplace_back(std::move(br));
        }
    }
    VD.branches.swap(kept);

    // Reseed new branch walks from endpoints inside the update region
    std::unordered_set<std::pair<int,int>, PairHash> visited_local;
    visited_local.reserve(region_edges.size());
    for (int i=0; i<N; ++i) {
        if (!in_region[i]) continue;
        if (!is_endpoint_[i]) continue;
        for (int nb_i : VD.global_adj[i]) {
            // if (!in_region[nb_i]) continue;
            auto br = walk_branch(i, nb_i, in_region, visited_local);
            if (!br.empty()) {
                const bool enough_vers = (static_cast<int>(br.size()) - 1) >= branch_min_vers_;
                if (enough_vers) VD.branches.emplace_back(std::move(br));
            }
        }
    }
    return 1;
}

bool ViewpointManager::branch_reduction() {
    VD.branches_simpl.clear();
    VD.branches_simpl.reserve(VD.branches.size());

    for (size_t b=0; b<VD.branches.size(); ++b) {
        const auto& br = VD.branches[b];
        if (br.size() < 2) continue;
        auto simpl = reduce_branch(br);
        if (simpl.size() < 2) continue;
        VD.branches_simpl.emplace_back(simpl);
    }

    build_reduced_skeleton();
    return 1;
}

bool ViewpointManager::viewpoint_sampling() {

    for (size_t i=0; i<VD.gskel_size; ++i) {
        auto& vertex = VD.global_skel[i];
        if (vertex.type_update || vertex.pos_update) vertex.vpts.clear();
        if (!vertex.vpts.empty()) continue; // already has viewpoints and is not cleared

        std::vector<Viewpoint> new_vpts = generate_viewpoint(vertex.vid);

        
    }

    return 1;
}

/* Helpers */

std::vector<Viewpoint> ViewpointManager::generate_viewpoint(int id) {
    std::vector<Viewpoint> vpts_out;
    auto& v = VD.global_skel[id];
    const auto& type = v.type;

    if (type == 1) {
        // leaf
        int id_adj = VD.global_adj[v.vid][0];
        auto& v_nb = VD.global_skel[id_adj];
        const Eigen::Vector3f p1 = v.position.getVector3fMap();
        const Eigen::Vector3f p2 = v_nb.position.getVector3fMap();
        Eigen::Vector3f dir = p2 - p1;
        if (dir.norm() < 1e-2f) return vpts_out;
        dir.normalize();
        return vpts_out;
    }
    else if (type == 2) {
        // branch 

        return vpts_out;
    }
    else if (type == 3) {
        // joint

        return vpts_out;
    }
    else {
        vpts_out.clear();
        return vpts_out;
    }
}

std::vector<int> ViewpointManager::walk_branch(int start_idx, int nb_idx, const std::vector<char>& allowed, std::unordered_set<std::pair<int,int>, PairHash>& visited_edges) {
    auto edge_key = [](int u, int v) {
        if (u > v) std::swap(u,v);
        return std::make_pair(u,v);
    };

    auto edge_seen = [&](int a, int b) {
        return visited_edges.count(edge_key(a,b)) != 0;
    };
    
    auto mark_edge = [&](int a, int b) {
        visited_edges.insert(edge_key(a,b));
    };

    std::vector<int> out_vids;
    out_vids.reserve(32);

    if (start_idx < 0 || nb_idx < 0) return out_vids; // invalid
    if (!allowed[start_idx] || !allowed[nb_idx]) return out_vids; // not allowed
    if (edge_seen(start_idx, nb_idx)) return out_vids; // already seen

    int prev = start_idx;
    int curr = nb_idx;

    out_vids.push_back(idx2vid_[start_idx]);

    while (true) {
        out_vids.push_back(idx2vid_[curr]);
        mark_edge(prev, curr);

        if (is_endpoint_[curr] && curr != start_idx) break; // found endpoint
        int next_idx = -1;

        for (int nb_i : VD.global_adj[curr]) {
            if (nb_i == prev) continue;
            if (!allowed[nb_i]) continue;
            if (!edge_seen(curr, nb_i)) {
                next_idx = nb_i;
                break;
            }
        }

        // fallback: any nb neq prev
        if (next_idx == -1) {
            for (int nb_i : VD.global_adj[curr]) {
                if (nb_i != prev) {
                    next_idx = nb_i;
                    break;
                }
            }
        }

        if (next_idx == -1) break; // no next found!

        prev = curr;
        curr = next_idx;
    }
    
    if (out_vids.size() < 2) {
        out_vids.clear();
    }

    return out_vids;
}




// reduction

std::vector<int> ViewpointManager::reduce_branch(const std::vector<int>& br_vids) {
    const int n = static_cast<int>(br_vids.size());
    if (n <= 2) return br_vids;

     // ---- thresholds (tune or move to cfg_) ----
    const float eps_dist_add  = 0.20f; // m: add if chord error > this
    const float eps_dist_drop = 0.12f; // m: drop if < this (hysteresis)
    const float angle_add_deg = 12.0f; // deg: add if turning angle > this
    const float angle_drop_deg= 7.0f;  // deg: drop if < this (hysteresis)

    auto point_to_seg_dist = [&](const Eigen::Vector3f& p, const Eigen::Vector3f& a, const Eigen::Vector3f& b) -> float {
        Eigen::Vector3f ab = b - a;
        float ab2 = ab.squaredNorm();
        if (ab2 <= 1e-12f) return (p-a).norm();
        float t = (p - a).dot(ab) / ab2;
        // t = std::clamp(t, 0.0f, 1.0f);
        t = std::min(std::max(t,0.0f), 1.0f);
        Eigen::Vector3f proj = a + t * ab;
        return (p - proj).norm();
    };

    auto turning_angle_deg = [&](const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c) -> float {
        Eigen::Vector3f u = b - a;
        float nu = u.norm();
        if (nu > 0) u /= nu;
        Eigen::Vector3f v = c - b;
        float nv = v.norm();
        if (nv > 0) v /= nv;
        float d = std::min(1.0f, std::max(-1.0f, u.dot(v)));
        return std::acos(d) * 180.0f / static_cast<float>(M_PI);
    };

    // Gather 3D positions for branch VIDs
    std::vector<Eigen::Vector3f> P(n);
    P.reserve(n);
    for (int i=0; i<n; ++i) {
        int idx = vid2idx_.at(br_vids[i]); // assumes every vid exists (obs?)
        const auto& q = VD.global_skel[idx].position;
        P[i] = q.getVector3fMap();
    }

    // Seeds at endpoints + high-curvature points
    std::vector<char> keep(n,0);
    std::vector<char> must_keep(n,0);
    keep.front() = keep.back() = 1;
    must_keep.front() = must_keep.back() = 1;

    for (int i=1; i<n-1; ++i) {
        float ang = turning_angle_deg(P[i-1], P[i], P[i+1]);
        if (ang >= angle_add_deg) {
            must_keep[i] = 1;
            keep[i] = 1;
        }
    }

    struct Seg { int a; int b; };
    std::vector<Seg> stack;
    stack.reserve(n);
    stack.push_back({0,n-1});

    auto farthest_index = [&](int i0, int i1, bool only_must_keep, float& out_d) -> int {
        int best_k = -1;
        float best_d = -1.0f;
        for (int k=i0+1; k<i1; ++k) {
            if (only_must_keep && !must_keep[k]) continue;
            float d = point_to_seg_dist(P[k], P[i0], P[i1]);
            if (d > best_d) {
                best_d = d;
                best_k = k;
            }
        }
        out_d = best_d;
        return best_k;
    };

    while (!stack.empty()) {
        auto [a, b] = stack.back();
        stack.pop_back();
        if (b - a <= 1) continue;

        float maxd = -1.0f;
        int split_k = farthest_index(a, b, true, maxd);  // must_keep_only = true
        if (split_k == -1) {
            // Could not split at a must_keep point
            split_k = farthest_index(a, b, false, maxd);
            if ((split_k != -1) && (maxd <= eps_dist_add)) {
                keep[a] = 1;
                keep[b] = 1;
                continue;
            }
        }

        if (split_k != -1) {
            keep[split_k] = 1;
            stack.push_back({a, split_k});
            stack.push_back({split_k, b});
        }
    }

    // Hysteresis drop for stability...
    for (int i=1; i<n-1; ++i) {
        if (keep[i]) {
            float ang = turning_angle_deg(P[i-1], P[i], P[i+1]);
            float err = point_to_seg_dist(P[i], P[i-1], P[i+1]);
            if (!must_keep[i] && ang < angle_drop_deg && err < eps_dist_drop) {
                keep[i] = 0;
            }
        }
    }

    // Build simplified branch VIDs
    std::vector<int> out;
    out.reserve(n);
    for (int i=0; i<n; ++i) {
        if (keep[i]) {
            out.push_back(br_vids[i]);
        }
    }

    if (static_cast<int>(out.size()) < 2) return br_vids; // never return degenerate...
    return out;
}

void ViewpointManager::build_reduced_skeleton() {
    if (VD.branches_simpl.empty()) {
        VD.reduced_skel.clear();
        return;
    }

    std::unordered_map<int, std::unordered_set<int>> adj_vids;
    adj_vids.reserve(VD.branches_simpl.size() * 4);

    auto add_edge = [&](int u, int v) {
        if (u == v) return;
        adj_vids[u].insert(v);
        adj_vids[v].insert(u);
    };

    for (const auto& br : VD.branches_simpl) {
        if (br.size() < 2) continue;
        for (size_t i=1; i<br.size(); ++i) {
            add_edge(br[i-1], br[i]);
        }
    }

    if (adj_vids.empty()) {
        VD.reduced_skel.clear();
        return;
    }

    std::vector<int> vids;
    vids.reserve(adj_vids.size());
    for (const auto& kv : adj_vids) {
        vids.push_back(kv.first);
    }
    std::sort(vids.begin(), vids.end());

    VD.reduced_skel.clear();
    VD.reduced_skel.reserve(vids.size());

    for (int vid : vids) {
        auto it_idx = vid2idx_.find(vid);
        if (it_idx == vid2idx_.end()) continue;

        const int gi = it_idx->second;
        const auto& src = VD.global_skel[gi];

        Vertex rv;
        rv.vid = vid;
        rv.position = src.position;
        rv.pos_update = false;
        rv.type_update = false;

        const auto& nset = adj_vids[vid];
        rv.nb_ids.assign(nset.begin(), nset.end());
        std::sort(rv.nb_ids.begin(), rv.nb_ids.end());

        const int deg = static_cast<int>(rv.nb_ids.size());
        rv.type = (deg == 1) ? 1 : (deg == 2) ? 2 : (deg > 2) ? 3 : 0;
        VD.reduced_skel.emplace_back(std::move(rv));
    }
}