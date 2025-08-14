/* 

Main Path Skeleton Guided Viewpoint Generation and Path Planning

This file contains the incremental point cloud incrementation

*/

// #include "osep_planning/planner_main.hpp"
#include "planner_main.hpp"

PathPlanner::PathPlanner(rclcpp::Node::SharedPtr node) : node_(node)
{
}

void PathPlanner::init() {
    RCLCPP_INFO(node_->get_logger(), "Initializing Module: Online Skeleton Guided Path Planner");
    /* Param */
    // Stuff from launch file (ToDo)...

    /* Data */
    GS.global_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
    GS.global_seen_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    GS.global_vertices_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

    local_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
    local_vertices.reset(new pcl::PointCloud<pcl::PointXYZ>);

    GP.global_vpts_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    GS.gskel_size = 0;
}

void PathPlanner::update_skeleton() {
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int i=0; i<(int)GS.global_vertices.size(); ++i) {
        GS.global_vertices[i].prev_type = GS.global_vertices[i].type;
        GS.global_vertices[i].updated = false;
    }

    skeleton_increment();
    graph_adj();
    mst();
    vertex_merge();
    prune_branches();
    smooth_vertex_positions();
    extract_branches();

    for (int i=0; i<(int)GS.global_vertices.size(); ++i) {
        if (GS.global_vertices[i].type != GS.global_vertices[i].prev_type) {
            GS.global_vertices[i].updated = true;
        }
    }

    GS.gskel_size = (int)GS.global_vertices.size(); // Update total number of vertices
    // RCLCPP_INFO(node_->get_logger(), "Global Skeleton Size: %d", GS.gskel_size);

    RCLCPP_INFO(node_->get_logger(), "[Skelton Update] Number of branches: %d", (int)GS.branches.size());

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_elapsed = t_end - t_start;
    RCLCPP_INFO(node_->get_logger(), "[Skeleton Update] Time Elapsed: %f seconds", t_elapsed.count());
}

void PathPlanner::plan_path() {
    if (GS.global_vertices.size() < 2) {
        RCLCPP_WARN(node_->get_logger(), "[Select Viewpoints] Too few vertices to plan.");
        return;
    }
    
    // auto t_start = std::chrono::high_resolution_clock::now();

    viewpoint_sampling();
    viewpoint_filtering();
    build_visibility_graph();

    int path_length = (int)GP.local_path.size();
    if (path_length < MAX_HORIZON) {
        auto dfs_t1 = std::chrono::high_resolution_clock::now();
        generate_path();
        ++dfs_plan_cnt;
        auto dfs_t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dfs_t = dfs_t2 - dfs_t1;
        mean_dfs_time += (dfs_t.count() - mean_dfs_time) / static_cast<double>(dfs_plan_cnt);
        RCLCPP_WARN(node_->get_logger(), "[Path Planning] Mean DFS time: %f over %d runs", mean_dfs_time, dfs_plan_cnt);
    }
    refine_path();

    int N_vpts = (int)GP.global_vpts.size();
    int N_vpts_visited = 0;
    for (auto &vp : GP.global_vpts) {
        if (vp.visited) ++N_vpts_visited;
    }

    RCLCPP_INFO(node_->get_logger(), "Number of viewpoint: %d", N_vpts);
    RCLCPP_INFO(node_->get_logger(), "Number of visited viewpoints %d", N_vpts_visited);
    RCLCPP_INFO(node_->get_logger(), "Current Local Path Lenght: %d", (int)GP.local_path.size());

    if (!GS.global_pts->points.empty()) {
        double current_coverage = static_cast<double>(GS.global_seen_cloud->points.size()) / static_cast<double>(GS.global_pts->points.size());
        RCLCPP_INFO(node_->get_logger(), "Current Coverage Percentage: %f", current_coverage);
    }

    // auto t_end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> t_elapsed = t_end - t_start;
    // RCLCPP_INFO(node_->get_logger(), "[Path Planning] Time Elapsed: %f seconds", t_elapsed.count());
}


/* Skeleton Updating*/
void PathPlanner::skeleton_increment() {
    if (!local_vertices || local_vertices->points.empty()) return;

    RCLCPP_INFO(node_->get_logger(), "Updating Global Skeleton...");
    // std::vector<int> ids_to_delete;

    for (auto &v : GS.prelim_vertices) {
        v.just_approved = false;
    }

    for (auto &pt : local_vertices->points) {
        Eigen::Vector3d ver(pt.x, pt.y, pt.z);
        bool matched = false;

        for (int i=0; i<(int)GS.prelim_vertices.size(); ++i) {
            auto &gver = GS.prelim_vertices[i];

            double sq_dist = (gver.position - ver).squaredNorm();
            if (sq_dist < fuse_dist_th*fuse_dist_th) {
                
                if (gver.freeze) {
                    matched = true;
                    break;
                }

                VertexLKF kf(kf_pn, kf_mn);
                kf.initialize(gver.position, gver.covariance);
                kf.update(ver);

                gver.position = kf.getState();
                gver.covariance = kf.getCovariance();
                gver.obs_count++;
                double trace = gver.covariance.trace();

                if (!gver.conf_check && trace < fuse_conf_th) {
                    gver.conf_check = true;
                    gver.just_approved = true;
                    gver.freeze = true; // Freeze vertex!
                    gver.unconfirmed_check = 0;
                }
                else {
                    gver.unconfirmed_check++;
                }
                
                // Mark ids as invalid and schedule for removal if not confident after a few runs...
                if (gver.unconfirmed_check > max_obs_wo_conf) {
                    // ids_to_delete.push_back(i);
                    gver.marked_for_deletion = true;
                }

                matched = true;
                break;
            }
        }

        if (!matched) {
            SkeletonVertex new_ver;
            new_ver.position = ver;
            new_ver.covariance = Eigen::Matrix3d::Identity();
            new_ver.obs_count = 1;
            new_ver.conf_check = false;
            new_ver.freeze = false;
            new_ver.just_approved = false;
            new_ver.unconfirmed_check = 0;
            GS.prelim_vertices.push_back(new_ver);
        }
    }

    // Delete points that did not pass the confidence check (reverse indexing for id consistency)
    // for (auto it = ids_to_delete.rbegin(); it != ids_to_delete.rend(); ++it) {
    //     GS.prelim_vertices.erase(GS.prelim_vertices.begin() + *it);
    // }

    // for (auto it = ids_to_delete.rbegin(); it != ids_to_delete.rend(); ++it) {
    //     if (*it >= 0 && *it < (int)GS.prelim_vertices.size()) {
    //         GS.prelim_vertices.erase(GS.prelim_vertices.begin() + *it);
    //     } else {
    //         RCLCPP_WARN(node_->get_logger(), "Skipping deletion of invalid index: %d", *it);
    //     }
    // }

    GS.prelim_vertices.erase(
        std::remove_if(
            GS.prelim_vertices.begin(),
            GS.prelim_vertices.end(),
            [](const SkeletonVertex& v) {
                return v.marked_for_deletion;
            }
        ),
        GS.prelim_vertices.end()
    );

    // Pass confident vertices to the global skeleton...
    GS.new_vertex_indices.clear();
    for (int i=0; i<(int)GS.prelim_vertices.size(); ++i) {
        auto &v = GS.prelim_vertices[i];
        if (v.just_approved && v.position.z() > gnd_th) {
            // GS.global_vertices.push_back(v);
            GS.global_vertices.emplace_back(std::move(v));
            GS.global_vertices_cloud->points.emplace_back(v.position.x(), v.position.y(), v.position.z());
            GS.new_vertex_indices.push_back((int)GS.global_vertices.size() - 1);
            v.just_approved = false;
            v.updated = true;
            v.spawned_vpts = false;
        }
    }
}

void PathPlanner::graph_adj() {
    if (!GS.global_vertices_cloud || GS.global_vertices_cloud->empty()) {
        RCLCPP_WARN(node_->get_logger(), "No points in global skeleton. Skipping graph adjacency rebuild.");
        return;
    }

    // Create a new adjacency list
    std::vector<std::vector<int>> new_adj(GS.global_vertices_cloud->size());

    pcl::KdTreeFLANN<pcl::PointXYZ> adj_tree;
    adj_tree.setInputCloud(GS.global_vertices_cloud);

    const int K = 10;          // Number of neighbors
    const float max_dist_th = 2.5 * fuse_dist_th; // Max distance for valid edges (meters)

    for (size_t i = 0; i < GS.global_vertices_cloud->size(); ++i) {
        std::vector<int> indices;
        std::vector<float> distances;

        int n_neighbors = adj_tree.nearestKSearch(GS.global_vertices_cloud->points[i], K, indices, distances);

        for (int j = 1; j < n_neighbors; ++j) { // Skip self (index 0)
            int nb_idx = indices[j];
            float dist_to_nb = (GS.global_vertices_cloud->points[i].getVector3fMap() - GS.global_vertices_cloud->points[nb_idx].getVector3fMap()).norm();

            if (dist_to_nb > max_dist_th) continue; // Too far or too close, skip

            bool is_good_neighbor = true;

            // Small geometric consistency check
            for (int k = 1; k < n_neighbors; ++k) {
                if (k == j) continue;

                int other_nb_idx = indices[k];
                float dist_nb_to_other = (GS.global_vertices_cloud->points[nb_idx].getVector3fMap() - GS.global_vertices_cloud->points[other_nb_idx].getVector3fMap()).norm();
                float dist_to_other = (GS.global_vertices_cloud->points[i].getVector3fMap() - GS.global_vertices_cloud->points[other_nb_idx].getVector3fMap()).norm();

                if (dist_nb_to_other < dist_to_nb && dist_to_other < dist_to_nb) {
                    is_good_neighbor = false;
                    break;
                }
            }

            if (is_good_neighbor) {
                new_adj[i].push_back(nb_idx);
                new_adj[nb_idx].push_back(i);
            }
        }
    }

    GS.global_adj = new_adj; // Replace old adjacency
}

void PathPlanner::mst() {
    int N_ver = GS.global_vertices.size();
    if (N_ver == 0 || GS.global_adj.empty()) {
        RCLCPP_WARN(node_->get_logger(), "Global skeleton is empty, cannot extract MST.");
        return;
    }

    std::vector<Edge> mst_edges;
    for (int i = 0; i < N_ver; ++i) {
        for (int nb : GS.global_adj[i]) {
            if (nb <= i) continue; // Avoid bi-directional check
            Eigen::Vector3d ver_i = GS.global_vertices[i].position;
            Eigen::Vector3d ver_nb = GS.global_vertices[nb].position;
            double weight = (ver_i - ver_nb).norm();
            mst_edges.push_back({i, nb, weight});
        }
    }

    std::sort(mst_edges.begin(), mst_edges.end());

    UnionFind uf(N_ver);
    std::vector<std::vector<int>> mst_adj(N_ver);

    for (const auto& edge : mst_edges) {
        if (uf.unite(edge.u, edge.v)) {
            mst_adj[edge.u].push_back(edge.v);
            mst_adj[edge.v].push_back(edge.u);
        }
    }

    GS.global_adj = std::move(mst_adj);

    // Identify leafs and joints
    graph_decomp();
}

void PathPlanner::vertex_merge() {
    int N_ver = GS.global_vertices.size();
    int new_ver = GS.new_vertex_indices.size();
    if (N_ver == 0 || new_ver == 0) return;
    if (static_cast<double>(new_ver) / static_cast<double>(N_ver) > 0.5) return;

    // if (N_ver < 5) return;

    auto is_joint = [&](int idx) {
        return std::find(GS.joints.begin(), GS.joints.end(), idx) != GS.joints.end();
    };

    std::set<int> new_set(GS.new_vertex_indices.begin(), GS.new_vertex_indices.end());
    std::set<int> to_delete;

    for (int j_new : GS.new_vertex_indices) {
        for (int i : GS.global_adj[j_new]) {
            if (i == j_new || to_delete.count(i) || to_delete.count(j_new)) {
                continue;
            }

            bool do_merge = false;

            if (is_joint(i) && is_joint(j_new)) {
                do_merge = true;
                GS.joints.erase(std::remove(GS.joints.begin(), GS.joints.end(), j_new), GS.joints.end());
            }

            double dist = (GS.global_vertices[i].position - GS.global_vertices[j_new].position).norm();
            if (!do_merge && dist < 0.5 * fuse_dist_th) {
                do_merge = true;
            }

            if (!do_merge) continue;

            merge_into(i, j_new); //merge j_new into i

            //Maybe: To generate new viewpoints for moved vertex??
            // GS.global_vertices[i].updated = true;

            to_delete.insert(j_new);
            break;
        }
    }

    if (to_delete.empty()) return;

    std::vector<int> del_idx(to_delete.begin(), to_delete.end());
    std::sort(del_idx.rbegin(), del_idx.rend());

    for (int idx : del_idx) {
        GS.global_vertices.erase(GS.global_vertices.begin() + idx);
        GS.global_vertices_cloud->points.erase(GS.global_vertices_cloud->points.begin() + idx);

        GS.global_adj.erase(GS.global_adj.begin() + idx);
        for (auto &nbrs : GS.global_adj) {
            nbrs.erase(std::remove(nbrs.begin(), nbrs.end(), idx), nbrs.end());
            for (auto &v : nbrs) {
                if (v > idx) --v;
            }
        }

        GS.new_vertex_indices.erase(std::remove(GS.new_vertex_indices.begin(), GS.new_vertex_indices.end(), idx), GS.new_vertex_indices.end());
        for (auto& id : GS.new_vertex_indices) {
            if (id > idx) --id;
        }
    }

    graph_decomp(); // update leafs and joints
}

void PathPlanner::prune_branches() {
    int N_ver = GS.global_vertices.size();
    int new_ver = GS.new_vertex_indices.size();
    if (N_ver == 0 || new_ver == 0) return;
    if (static_cast<double>(new_ver) / static_cast<double>(N_ver) > 0.5) return;
    
    
    std::unordered_set<int> joint_set(GS.joints.begin(), GS.joints.end());
    std::unordered_set<int> new_set(GS.new_vertex_indices.begin(), GS.new_vertex_indices.end());
    std::vector<bool> to_delete(N_ver, false); // Mask for deleting

    for (int leaf : GS.leafs) {
        if (!new_set.count(leaf)) continue; // Not a new leaf

        if (GS.global_adj[leaf].size() != 1) continue; // Sanity check

        int nb = GS.global_adj[leaf][0];

        // If leaf nb is a joint -> remove leaf...
        if (joint_set.count(nb)) {
            to_delete[leaf] = true;
        }
    }

    std::vector<int> del_idx;
    for (int i=0; i<N_ver; ++i) {
        if (to_delete[i]) {
            del_idx.push_back(i);
        }
    }

    if (del_idx.empty()) return;

    std::sort(del_idx.rbegin(), del_idx.rend());

    for (int idx : del_idx) {
        GS.global_vertices.erase(GS.global_vertices.begin() + idx);
        GS.global_vertices_cloud->points.erase(GS.global_vertices_cloud->points.begin() + idx);
        
        GS.global_adj.erase(GS.global_adj.begin() + idx);
        for (auto& nbrs : GS.global_adj) {
            nbrs.erase(std::remove(nbrs.begin(), nbrs.end(), idx), nbrs.end());
            for (auto &v : nbrs) {
                if (v > idx) --v;
            }
        }
        
        GS.new_vertex_indices.erase(std::remove(GS.new_vertex_indices.begin(), GS.new_vertex_indices.end(), idx), GS.new_vertex_indices.end());
        for (auto& id : GS.new_vertex_indices) {
            if (id > idx) --id;
        }
    }

    graph_decomp();
}

void PathPlanner::smooth_vertex_positions() {
    if (GS.global_vertices.empty() || GS.global_adj.empty()) return;

    size_t N = GS.global_vertices.size();
    std::vector<Eigen::Vector3d> new_positions(N);
  
    // 1) compute the new positions, but only for those with smooth_iters_left > 0
    for (size_t i = 0; i < N; ++i) {
      auto &v    = GS.global_vertices[i];
      auto &nbrs = GS.global_adj[i];
  
      // never move joints or leaves
      if (v.type == 1 || v.type == 3 || nbrs.size() < 2) {
        new_positions[i] = v.position;
        continue;
      }
  
      // only smooth if this vertex still has budget
      if (v.smooth_iters_left > 0) {
        Eigen::Vector3d avg = Eigen::Vector3d::Zero();
        for (int j : nbrs) {
          avg += GS.global_vertices[j].position;
        }
        avg /= double(nbrs.size());
  
        double blend = 0.3;  // how strongly to pull toward the average
        new_positions[i] = (1.0 - blend)*v.position + blend*avg;
  
        // consume one smoothing iteration
        --v.smooth_iters_left;
      }
      else {
        new_positions[i] = v.position;
      }
    }
  
    // 2) write them back in place (so you don't lose any other vertex state)
    for (size_t i = 0; i < N; ++i) {
      GS.global_vertices[i].position = new_positions[i];
    }
  
    // 3) update your point cloud
    GS.global_vertices_cloud->clear();
    for (const auto &v : GS.global_vertices) {
      GS.global_vertices_cloud->points.emplace_back(
        v.position.x(), v.position.y(), v.position.z()
      );
    }
}

void PathPlanner::extract_branches() {
    const int min_branch_length = 5;
    int N = (int)GS.global_vertices.size();
    if (N == 0) return;

    GS.branches.clear();

    std::set<std::pair<int,int>> visited_edges;

    // Helper function for determining endpoints
    auto is_endpoint = [&](int vidx) {
        return GS.global_vertices[vidx].type == 1 || GS.global_vertices[vidx].type == 3;
        // return std::find(GS.leafs.begin(), GS.leafs.end(), vidx) != GS.leafs.end() ||
        //        std::find(GS.joints.begin(), GS.joints.end(), vidx) != GS.joints.end();
    };

    // For every endpoint: walk each unvisited nb
    for (int endp : GS.leafs) {
        for (int nb : GS.global_adj[endp]) {
            int a = endp;
            int b = nb;
            if (a > b) std::swap(a,b); // ensure that edge a->b and b->a is interpreted the same
            if (visited_edges.count({a,b})) continue; // edge already walked

            std::vector<int> branch;
            branch.push_back(endp);

            int prev = endp;
            int curr = nb;

            while (true) {
                branch.push_back(curr);

                if (is_endpoint(curr) && curr != endp) break; // stop if another endpoint is hit

                int next = -1;

                for (int nn : GS.global_adj[curr]) {
                    if (nn != prev) {
                        next = nn;
                        break;
                    }
                }

                if (next == -1) break; // dead end

                prev = curr;
                curr = next;
            }

            for (int i=1; i<(int)branch.size(); ++i) {
                int x = branch[i-1];
                int y = branch[i];
                visited_edges.insert({x,y});
            }

            GS.branches.push_back(branch);
        }
    }

    // repeat for joints to catch joint-joint bounded branches
    for (int endp : GS.joints) {
        for (int nb : GS.global_adj[endp]) {
            int a = endp;
            int b = nb;
            if (a > b) std::swap(a,b);
            if (visited_edges.count({a,b})) continue;

            std::vector<int> branch;
            branch.push_back(endp);

            int prev = endp;
            int curr = nb;

            while(true) {
                branch.push_back(curr);
                if (is_endpoint(curr) && curr != endp) break;

                int next = -1;
                for (int nn : GS.global_adj[curr]) {
                    if (nn != prev) {
                        next = nn;
                        break;
                    }
                }

                if (next == -1) break;

                prev = curr;
                curr = next;
            }

            for (int i=1; i<(int)branch.size(); ++i) {
                int x = branch[i-1];
                int y = branch[i];
                visited_edges.insert({x,y});
            }
            GS.branches.push_back(branch);
        }
    }

    // sort branches for ascending order
    std::sort(GS.branches.begin(), GS.branches.end(), 
        [](const std::vector<int>& a, const std::vector<int>&b) {
            return a.size() < b.size();
        }
    );

    // prune small branches
    GS.branches.erase(
        std::remove_if(
            GS.branches.begin(), GS.branches.end(),
            [min_branch_length](const std::vector<int>& branch) {
                return branch.size() < min_branch_length;
            }
        ),
        GS.branches.end()
    );
}

void PathPlanner::graph_decomp() {
    /* Assigns vertex type */
    GS.joints.clear();
    GS.leafs.clear();

    for (int i=0; i<(int)GS.global_vertices.size(); ++i) {
        // int prev_type = GS.global_vertices[i].type;
        int degree = GS.global_adj[i].size();
        int new_type = 0;
        
        if (degree == 1) { // Leaf
            GS.leafs.push_back(i);
            // GS.global_vertices[i].type = 1;
            new_type = 1;
        }

        else if (degree == 2) { // branch
            // GS.global_vertices[i].type = 2;
            new_type = 2;
        }

        else if (degree > 2) { // joint
            GS.joints.push_back(i);
            // GS.global_vertices[i].type = 3;
            new_type = 3;
        }

        GS.global_vertices[i].type = new_type;

        // Mark updated vertices if their type is updated in this iteration
        // if (GS.global_vertices[i].updated) continue;
        // GS.global_vertices[i].updated = GS.global_vertices[i].updated || (GS.global_vertices[i].type != prev_type);
    }
}

void PathPlanner::merge_into(int id_keep, int id_del) {
    auto &Vi = GS.global_vertices[id_keep];
    auto &Vj = GS.global_vertices[id_del];

    int tot = Vi.obs_count + Vi.obs_count;
    Vi.position = (Vi.position * Vi.obs_count + Vj.position * Vj.obs_count) / tot;
    Vi.obs_count = tot;

    for (int nb: GS.global_adj[id_del]) {
        if (nb == id_keep) continue;
        auto &nbs_i = GS.global_adj[id_keep];
        if (std::find(nbs_i.begin(), nbs_i.end(), nb) == nbs_i.end()) {
            nbs_i.push_back(nb); // if nb of id_del is not already nb to id_keep
        }

        auto &nbs_nb = GS.global_adj[nb];
        std::replace(nbs_nb.begin(), nbs_nb.end(), id_del, id_keep);
    }
}


/* Viewpoint Generation and Path Planning */

void PathPlanner::viewpoint_sampling() {
    // Collect pointers to old viewpoints for removal
    std::unordered_set<const Viewpoint*> to_remove;

    for (int i = 0; i < GS.gskel_size; ++i) {
        auto &vertex = GS.global_vertices[i];

        // Skip if already spawned viewpoints OR not update OR type 0
        if (vertex.spawned_vpts || !vertex.updated || vertex.type == 0) continue;
        // if (!vertex.updated || vertex.type == 0) continue;

        if (vertex.type == 2 && vertex.smooth_iters_left != 0) continue; // Spawn viewpoints when smoothing is complete

        // Record all currently assigned viewpoints for removal
        if (vertex.spawned_vpts) {
            for (Viewpoint* vp : vertex.assigned_vpts) {
                to_remove.insert(vp);
            }
        }

        vertex.assigned_vpts.clear();
    }

    // Remove any old viewpoints that were cleared above
    if (!to_remove.empty()) {
        GP.global_vpts.remove_if([&](const Viewpoint &vp) {
            return to_remove.count(&vp) > 0;
        });
    }

    for (int i=0; i<GS.gskel_size; ++i) {
        auto &vertex = GS.global_vertices[i];
        if (vertex.spawned_vpts || !vertex.updated || vertex.type == 0) continue;

        std::vector<Viewpoint> new_vpts = generate_viewpoint(i);
        vertex.spawned_vpts = true;
        for (auto &vp : new_vpts) {
            GP.global_vpts.emplace_back(std::move(vp));
            Viewpoint* vp_ptr = &GP.global_vpts.back();  // get pointer to stable list element
            vertex.assigned_vpts.push_back(vp_ptr);      // store pointer directly
        }
    }

        // Generate new viewpoints
        // std::vector<Viewpoint> new_vpts = generate_viewpoint(i);
        // vertex.spawned_vpts = true;

        // if (new_vpts.empty())
        //     continue;

        // // Append them into a std::list for pointer stability
        // for (auto &vp : new_vpts) {
        //     GP.global_vpts.emplace_back(std::move(vp));
        // }

    // Remove any old viewpoints that were cleared above
    

    // if (!to_remove.empty()) {
    //     GP.global_vpts.remove_if([&](const Viewpoint &vp) {
    //         return to_remove.count(const_cast<Viewpoint*>(&vp)) > 0;
    //     });
    // }
}

void PathPlanner::viewpoint_filtering() {
    if (GS.global_pts->empty()) {
        GP.global_vpts.clear();
        RCLCPP_WARN(node_->get_logger(), "No VoxelMap - Skipping Viewpoint Generation");
        return;
    }

    for (auto &v : GS.global_vertices) {
        v.assigned_vpts.clear();
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> voxel_tree;
    voxel_tree.setInputCloud(GS.global_pts);

    for (auto it = GP.global_vpts.begin(); it != GP.global_vpts.end(); /**/) {
        Viewpoint &vp = *it;

        if (!viewpoint_check(vp, voxel_tree)) {
            it = GP.global_vpts.erase(it);
            continue;
        }

        for (auto jt = std::next(it); jt != GP.global_vpts.end(); /**/) {
            if (viewpoint_similarity(vp, *jt)) {
                vp.position = 0.5 * (vp.position + jt->position);
                vp.orientation = vp.orientation.slerp(0.5, jt->orientation);
                jt = GP.global_vpts.erase(jt);
            }
            else {
                ++jt;
            }
        }

        int vid = vp.corresp_vertex_id;
        GS.global_vertices[vid].assigned_vpts.push_back(&vp);

        ++it; 
    }

    // Clean up potential dangling pointerss
    {
        std::unordered_set<Viewpoint*> valid;
        valid.reserve(GP.global_vpts.size());
        for (auto &vp : GP.global_vpts) valid.insert(&vp);

        auto &path = GP.local_path;
        path.erase(
          std::remove_if(path.begin(), path.end(),
            [&](Viewpoint *p){ return valid.count(p) == 0; }),
          path.end()
        );
    }
}

void PathPlanner::build_visibility_graph() {
    if (GP.global_vpts.empty()) return;

    GP.global_vpts_cloud->points.clear();

    std::vector<Viewpoint*> all_vpts;
    pcl::PointXYZ vp_pt;
    for (auto& vp : GP.global_vpts) {
        vp.adj.clear();
        all_vpts.push_back(&vp);
        vp_pt.x = vp.position.x();
        vp_pt.y = vp.position.y();
        vp_pt.z = vp.position.z();
        GP.global_vpts_cloud->points.push_back(vp_pt);
    }

    GP.global_vpts_tree.setInputCloud(GP.global_vpts_cloud);
    int N = all_vpts.size();
    std::vector<int> nb_ids;
    std::vector<float> nb_dist2;

    double total_edge_len = 0.0;
    int edge_count = 0;

    for (int i = 0; i < N; ++i) {
        nb_ids.clear();
        nb_dist2.clear();
        pcl::PointXYZ& search_point = GP.global_vpts_cloud->points[i];
        int n_nbs = GP.global_vpts_tree.radiusSearch(search_point, visibility_graph_radius, nb_ids, nb_dist2);

        for (int jj = 0; jj < n_nbs; ++jj) {
            int j = nb_ids[jj];
            if (j <= i) continue;  // only add each edge once

            Viewpoint* vi = all_vpts[i];
            Viewpoint* vj = all_vpts[j];

            // Check line-of-sight
            if (corridor_obstructed(vi->position, vj->position))
                continue;

            // Gabriel test: no other point inside the circle with ij as diameter
            Eigen::Vector3d mid = 0.5 * (vi->position + vj->position);
            double r2 = nb_dist2[jj] * 0.25;  // (d/2)^2
            bool keep_edge = true;

            for (int kk = 0; kk < n_nbs; ++kk) {
                int k = nb_ids[kk];
                if (k == i || k == j) continue;
                Eigen::Vector3d vk = all_vpts[k]->position;
                if ((vk - mid).squaredNorm() < r2) {
                    keep_edge = false;
                    break;
                }
            }

            if (!keep_edge)
                continue;

            // // Avoid connecting antiparallel viewpoints
            Eigen::Vector3d fwd_i = vi->orientation * Eigen::Vector3d::UnitX();
            Eigen::Vector3d fwd_j = vj->orientation * Eigen::Vector3d::UnitX();
            Eigen::Vector2d dir_i(fwd_i.x(), fwd_i.y());
            Eigen::Vector2d dir_j(fwd_j.x(), fwd_j.y());
            dir_i.normalize();
            dir_j.normalize();
            // cos(120) = cos(2pi/3) = -0.5
            double dot = dir_i.dot(dir_j);
            if (dot < -0.5) {
                continue;
            }

            // Add edge
            vi->adj.push_back(vj);
            vj->adj.push_back(vi);

            double length = std::sqrt(nb_dist2[jj]);
            total_edge_len += length;
            ++edge_count;
        }
    }

    // Compute mean edge distance for normalization
    mean_edge_dist = total_edge_len / static_cast<double>(edge_count);
    RCLCPP_INFO(node_->get_logger(), "Mean Edge Distance: %f", mean_edge_dist);
}

void PathPlanner::generate_path() {
    if (GP.global_vpts.empty()) return;

    // --- Determine starting viewpoint
    Viewpoint* start = nullptr;

    if (!GP.local_path.empty()) {
        start = GP.local_path.back();
    } 
    else {
        Eigen::Vector3d ref_pos = pose.position;
        double    best_dist = std::numeric_limits<double>::max();
        for (auto& vp : GP.global_vpts) {
            // if (vp.visited || vp.in_path) continue;
            if (vp.visited) continue;
            double d2 = (vp.position - ref_pos).squaredNorm();
            if (d2 < best_dist) {
                best_dist = d2;
                start = &vp;
            }
        }
    }
    if (!start) return;

    // --- Compute how many new slots in path left to fill
    int slots_left = MAX_HORIZON - static_cast<int>(GP.local_path.size());
    if (slots_left <= 0) return;

    // --- Run DFS to get a short‐horizon extension
    std::vector<Viewpoint*> dfs_path = dfs_run(start, DFS_MAX_DEPTH);

    if (mean_dfs_reward < REPLAN_THRESH) {
        RCLCPP_WARN(node_->get_logger(), "Triggering fallback due to low DFS reward (%.2f)", mean_dfs_reward);
        trigger_replan();
    }

    // --- Assemble new local path, keeping existing tail
    std::vector<Viewpoint*> new_local = GP.local_path;
    new_local.reserve(GP.local_path.size() + dfs_path.size());
    for (auto* vp : dfs_path) {
        if (slots_left <= 0) break;
        if (vp->in_path) continue;
        new_local.push_back(vp);
        vp->in_path = true;
        --slots_left;
    }

    // --- Enforce length and swap in new plan (should not happen...)
    if ((int)new_local.size() > MAX_HORIZON) {
        new_local.resize(MAX_HORIZON);
    }

    // --- Update path
    GP.local_path = std::move(new_local);
    RCLCPP_WARN(node_->get_logger(), "New path length: %zu", GP.local_path.size());
}

void PathPlanner::refine_path() {
    // Purge dangling pointers as before
    {
        std::unordered_set<Viewpoint*> valid;
        valid.reserve(GP.global_vpts.size());
        for (auto &vp : GP.global_vpts) valid.insert(&vp);

        auto &path = GP.local_path;
        path.erase(
          std::remove_if(path.begin(), path.end(),
            [&](Viewpoint *p){ return valid.count(p) == 0; }),
          path.end()
        );
    }

    // Take the filtered path and re-validate connectivity & occlusion in original order
    std::vector<Viewpoint*> prev = std::move(GP.local_path);
    GP.local_path.clear();

    Eigen::Vector3d last_pos = pose.position;
    Viewpoint*    last_vp  = nullptr;

    for (Viewpoint* vp : prev) {
        // Check connectivity (if not the very first)
        if (last_vp) {
            auto &nbrs = last_vp->adj;
            if (std::find(nbrs.begin(), nbrs.end(), vp) == nbrs.end()) {
                vp->in_path = false;
                continue;  // drop it
            }
        }

        // Check occlusion
        if (corridor_obstructed(last_pos, vp->position)) {
            vp->in_path = false;
            continue;  // drop it
        }

        // If we reach here, vp is still good—keep it in order
        GP.local_path.push_back(vp);
        last_pos = vp->position;
        last_vp  = vp;
    }

    // Any dropped vps get their in_path cleared
    for (Viewpoint* vp : prev) {
        if (std::find(GP.local_path.begin(), GP.local_path.end(), vp)
            == GP.local_path.end())
        {
            vp->in_path = false;
        }
    }
}

void PathPlanner::trigger_replan() {
    // trigger replan if mean dfs reward is too low
    double best_leaf_score = 0.0;
    int next_leaf = -1;

    // Clear current path
    // for (auto &vp : GP.local_path) {
    //     vp->in_path = false;
    // }
    // GP.local_path.clear();
    
    for (int leaf : GS.leafs) {

        SkeletonVertex& ver = GS.global_vertices[leaf];
        double leaf_score = 0.0; 
        for (auto& vp : ver.assigned_vpts) {
            if (!vp->visited) {
                leaf_score += 1.0;
            }
        }
        leaf_score /= static_cast<double>(ver.assigned_vpts.size());

        if (leaf_score > best_leaf_score && leaf_score > 0.0) {
            best_leaf_score = leaf_score;
            next_leaf = leaf;
        }
    }

    if (next_leaf < 0) {
        RCLCPP_WARN(node_->get_logger(), "Could not find unvisited leaf!");
        return;
    }

    std::vector<Viewpoint*> new_local_start;
    double best_distance = std::numeric_limits<double>::max();
    Viewpoint* best_viewpoint = nullptr;
    for (Viewpoint* vp : GS.global_vertices[next_leaf].assigned_vpts) {
        if (vp->visited || vp->adj.empty()) continue;
        double dist = (vp->position - pose.position).norm();
        if (dist < best_distance) {
            best_distance = dist;
            best_viewpoint = vp;
        }
        // new_local_start.push_back(vp);
        // break;
    }

    // Clear current path
    for (auto &vp : GP.local_path) {
        vp->in_path = false;
    }
    GP.local_path.clear();
    
    // Set new start
    new_local_start.push_back(best_viewpoint);
    if (new_local_start.empty()) {
        RCLCPP_WARN(node_->get_logger(), "Selected fallback leaf has no usable viewpoints!");
        return;
    }

    RCLCPP_WARN(node_->get_logger(), "Fallback to leaf vertex %d with score %.2f", next_leaf, best_leaf_score);
    GP.local_path = std::move(new_local_start);
}

int PathPlanner::find_branch_index(int id) {
    for (int i=0; i<(int)GS.branches.size(); ++i) {
        auto it = std::find(GS.branches[i].begin(), GS.branches[i].end(), id);
        if (it != GS.branches[i].end()) {
            return i;
        }
    }
    return -1;
}

std::vector<Viewpoint*> PathPlanner::dfs_run(Viewpoint* start, const int max_depth) {
    double total_reward = 0.0;
    double reward_samples = 0.0;

    std::vector<Viewpoint*> best_path;
    double best_score = -std::numeric_limits<double>::infinity();
    std::vector<Viewpoint*> current_path;

    // Precompute optimistic per-step reward
    double max_step_reward = COVERAGE_W + LEAF_W + BRANCH_W;

    std::unordered_set<Viewpoint*> dfs_visited;

    std::function<void(Viewpoint*, int, double)> dfs =
        [&](Viewpoint* node, int depth, double score) {
            dfs_visited.insert(node);
            current_path.push_back(node);

            int remaining = max_depth - depth;
            if (score + remaining * max_step_reward <= best_score) {
                current_path.pop_back();
                dfs_visited.erase(node);
                return;
            }

            // At depth limit or no neighbors
            if (depth == max_depth || node->adj.empty()) {
                if (score > best_score) {
                    best_score = score;
                    best_path = current_path;
                }
                current_path.pop_back();
                dfs_visited.erase(node);
                return;
            }

            // Collect neighbors and compute rewards
            std::vector<std::pair<Viewpoint*, double>> neighbors;
            // neighbors.reserve(node->adj.size());
            for (auto* nb : node->adj) {
                // if (std::find(current_path.begin(), current_path.end(), nb) != current_path.end())
                //     continue;
                if (dfs_visited.count(nb)) continue;
                double reward = dfs_reward(node, nb);
                total_reward += reward;
                ++reward_samples;
                neighbors.emplace_back(nb, reward);
            }

            // Backtracking if no neighbors found (and not at root)
            if (neighbors.empty() && depth > 0) {
                for (int backtrack_idx = static_cast<int>(current_path.size()) - 2; backtrack_idx >= 0; --backtrack_idx) {
                    Viewpoint* ancestor = current_path[backtrack_idx];
                    neighbors.emplace_back(ancestor, 0.0);
                    break;
                    // if (!dfs_visited.count(ancestor)) {
                    //     neighbors.emplace_back(ancestor, 0.0);
                    //     break;  // stop at the first available ancestor
                    // }
                }
            }

            // Beam search: take top-K
            int limit = std::min(BEAM_WIDTH, (int)neighbors.size());
            if (!neighbors.empty()) {
                std::partial_sort(
                    neighbors.begin(), neighbors.begin() + limit, neighbors.end(),
                    [](auto &a, auto &b) { return a.second > b.second; }
                );
            }

            // Recurse on selected neighbors
            for (int i = 0; i < limit; ++i) {
                Viewpoint* next = neighbors[i].first;
                double rwd = neighbors[i].second;
                // current_path.push_back(next);
                dfs(next, depth + 1, score + rwd);
                // current_path.pop_back();
            }

            // Consider stopping here
            if (score > best_score) {
                best_score = score;
                best_path = current_path;
            }

            current_path.pop_back();
            dfs_visited.erase(node);
        };

    current_path.push_back(start);
    dfs(start, 0, 0.0);

    // Remove the start viewpoint
    if (!best_path.empty() && best_path.front() == start) {
        best_path.erase(best_path.begin());
    }

    mean_dfs_reward = (reward_samples > 0) ? total_reward / reward_samples : 0.0;
    RCLCPP_WARN(node_->get_logger(), "MEAN DFS REWARD: %f", mean_dfs_reward);
    return best_path;
}

double PathPlanner::dfs_reward(Viewpoint* node, Viewpoint* nb) {
    score_viewpoint(nb);
    double cover_r = nb->score;
    
    double leaf_r = (GS.global_vertices[nb->corresp_vertex_id].type==1 && !nb->visited) ? 1.0 : 0.0;

    double branch_r = 0.0;
    int node_branch_id = find_branch_index(node->corresp_vertex_id);
    int nb_branch_id = find_branch_index(nb->corresp_vertex_id);
    if (nb_branch_id >= 0 && !nb->visited) {
        double branch_value = 0.0;
        for (int id : GS.branches[nb_branch_id]) {
            for (auto& vp : GS.global_vertices[id].assigned_vpts) {
                if (!vp->visited) {
                    branch_value += 1.0;
                } 
            }
        }
        if ((int)GS.branches[nb_branch_id].size() > 0) {
            branch_value = std::pow(branch_value / static_cast<double>(GS.branches[nb_branch_id].size()), 3.0);
        }
        // branch_r = node_branch_id == nb_branch_id ? branch_value : 0.0;
        double branch_continuity = (node_branch_id == nb_branch_id && node_branch_id >= 0) ? 0.5 : 0.0;
        branch_r = branch_value + branch_continuity;
    }

    double revisit_pen = nb->visited ? 1.0 : 0.0;
    double dist_pen = (node->position - nb->position).norm() / mean_edge_dist;
    
    double reward = cover_r * COVERAGE_W + leaf_r * LEAF_W + branch_r * BRANCH_W - dist_pen * DIST_W - revisit_pen * REVISIT_W;
    // RCLCPP_WARN(node_->get_logger(), "DFS REWARD: %f", reward);
    return reward;
}

std::vector<Viewpoint> PathPlanner::generate_viewpoint(int id) {
    std::vector<Viewpoint> output_vps;

    if (GS.global_vertices[id].type == 1) {
        // leaf
        int id_adj = GS.global_adj[id][0];
        const Eigen::Vector3d p1 = GS.global_vertices[id].position;
        const Eigen::Vector3d p2 = GS.global_vertices[id_adj].position;
        Eigen::Vector3d dir = p2 - p1;

        if (dir.norm() < 1e-2) return output_vps;
        dir.normalize();
        
        Eigen::Vector2d dir_xy = dir.head<2>();
        if (dir_xy.norm() > 0.5) {
            dir_xy.normalize();
            Eigen::Vector3d u1(-dir_xy.y(), dir_xy.x(), 0.0);
            Eigen::Vector3d u2(dir_xy.y(), -dir_xy.x(), 0.0);
            Eigen::Vector3d u3(-dir_xy.x(), -dir_xy.y(), 0.0);
            Eigen::Vector3d u4 = (u1 + u3).normalized();
            Eigen::Vector3d u5 = (u2 + u3).normalized();
            double d1 = distance_to_free_space(p1, u1);
            double d2 = distance_to_free_space(p1, u2);
            double d3 = distance_to_free_space(p1, u3);
            double d4 = distance_to_free_space(p1, u4);
            double d5 = distance_to_free_space(p1, u5);
            std::vector<double> dists = {d1, d2, d3, d4, d5};
            std::vector<Eigen::Vector3d> dirs = {u1, u2, u3, u4, u5};
            output_vps = vp_sample(p1, dirs, dists, id);
        }
        else {
            Eigen::Vector2d to_drone_xy = (pose.position.head<2>() - p1.head<2>());
            if (to_drone_xy.norm() < 1e-2) return output_vps;
            to_drone_xy.normalize();
            Eigen::Vector3d u1(to_drone_xy.x(), to_drone_xy.y(), 0.0);
            Eigen::Vector3d u2 = -u1;
            Eigen::Vector3d u3(-u1.y(), u1.x(), 0.0);
            Eigen::Vector3d u4(u1.y(), -u1.x(), 0.0);
            double d1 = distance_to_free_space(p1, u1);
            double d2 = distance_to_free_space(p1, u2);
            double d3 = distance_to_free_space(p1, u3);
            double d4 = distance_to_free_space(p1, u4);
            std::vector<double> dists = {d1, d2, d3, d4};
            std::vector<Eigen::Vector3d> dirs = {u1, u2, u3, u4};
            output_vps = vp_sample(p1, dirs, dists, id);
        }
    }

    if (GS.global_vertices[id].type == 2) {
        // branch
        int id_adj1 = GS.global_adj[id][0];
        int id_adj2 = GS.global_adj[id][1];
        const Eigen::Vector3d p1 = GS.global_vertices[id].position;
        const Eigen::Vector3d p2_1 = GS.global_vertices[id_adj1].position;
        const Eigen::Vector3d p2_2 = GS.global_vertices[id_adj2].position;
        Eigen::Vector3d dir = p2_1 - p2_2;

        if (dir.norm() < 1e-2) return output_vps;

        dir.normalize();
        Eigen::Vector2d dir_xy = dir.head<2>();
        if (dir_xy.norm() > 0.2) {
            dir_xy.normalize();
            Eigen::Vector3d u1(-dir_xy.y(), dir_xy.x(), 0.0);
            Eigen::Vector3d u2(dir_xy.y(), -dir_xy.x(), 0.0);
            // Eigen::Vector3d u3(dir_xy.x(), dir_xy.y(), 0.0);
            // Eigen::Vector3d u4 = -u3;
            double d1 = distance_to_free_space(p1, u1);
            double d2 = distance_to_free_space(p1, u2);
            // double d3 = distance_to_free_space(p1, u3);
            // double d4 = distance_to_free_space(p1, u4);
            // std::vector<double> dists = {d1, d2, d3, d4};
            std::vector<double> dists = {d1, d2};
            // std::vector<Eigen::Vector3d> dirs = {u1, u2, u3, u4};
            std::vector<Eigen::Vector3d> dirs = {u1, u2};
            output_vps = vp_sample(p1, dirs, dists, id);
        }
        else {
            Eigen::Vector2d to_drone_xy = (pose.position.head<2>() - p1.head<2>());
            if (to_drone_xy.norm() < 1e-2) return output_vps;
            to_drone_xy.normalize();
            Eigen::Vector3d u1(to_drone_xy.x(), to_drone_xy.y(), 0.0);
            Eigen::Vector3d u2 = -u1;
            Eigen::Vector3d u3(-u1.y(), u1.x(), 0.0);
            Eigen::Vector3d u4(u1.y(), -u1.x(), 0.0);
            double d1 = distance_to_free_space(p1, u1);
            double d2 = distance_to_free_space(p1, u2);
            double d3 = distance_to_free_space(p1, u3);
            double d4 = distance_to_free_space(p1, u4);
            std::vector<double> dists = {d1, d2, d3, d4};
            std::vector<Eigen::Vector3d> dirs = {u1, u2, u3, u4};
            output_vps = vp_sample(p1, dirs, dists, id);
        }
    }

    if (GS.global_vertices[id].type == 3) {
        return output_vps; // TEST - DROP JUNCTION VPTS
        
        // joint
        const Eigen::Vector3d p1 = GS.global_vertices[id].position;
        int N_nbs = GS.global_adj[id].size();
        std::vector<int> nbs = GS.global_adj[id];

        for (int i=0; i<N_nbs; ++i) {
            Eigen::Vector3d p2_1 = GS.global_vertices[nbs[i]].position;
            for (int j=i+1; j<N_nbs; ++j) {
                Eigen::Vector3d p2_2 = GS.global_vertices[nbs[j]].position;
                Eigen::Vector3d dir = p2_1 - p2_2;
                if (dir.norm() < 1e-2) continue;
                dir.normalize();

                Eigen::Vector2d dir_xy = dir.head<2>();
                if (dir_xy.norm() > 0.5) {
                    dir_xy.normalize();
                    Eigen::Vector3d u1(-dir_xy.y(), dir_xy.x(), 0.0);
                    Eigen::Vector3d u2(dir_xy.y(), -dir_xy.x(), 0.0);
                    double d1 = distance_to_free_space(p1, u1);
                    double d2 = distance_to_free_space(p1, u2);
                    std::vector<double> dists = {d1, d2};
                    std::vector<Eigen::Vector3d> dirs = {u1, u2};
                    auto vps = vp_sample(p1, dirs, dists, id);
                    output_vps.insert(output_vps.end(), vps.begin(), vps.end());
                }
                else {
                    Eigen::Vector2d to_drone_xy = (pose.position.head<2>() - p1.head<2>());
                    if (to_drone_xy.norm() < 1e-2) return output_vps;
                    to_drone_xy.normalize();
                    Eigen::Vector3d u1(to_drone_xy.x(), to_drone_xy.y(), 0.0);
                    Eigen::Vector3d u2 = -u1;
                    Eigen::Vector3d u3(-u1.y(), u1.x(), 0.0);
                    Eigen::Vector3d u4(u1.y(), -u1.x(), 0.0);
                    double d1 = distance_to_free_space(p1, u1);
                    double d2 = distance_to_free_space(p1, u2);
                    double d3 = distance_to_free_space(p1, u3);
                    double d4 = distance_to_free_space(p1, u4);
                    std::vector<double> dists = {d1, d2, d3, d4};
                    std::vector<Eigen::Vector3d> dirs = {u1, u2, u3, u4};
                    auto vps = vp_sample(p1, dirs, dists, id);
                    output_vps.insert(output_vps.end(), vps.begin(), vps.end());
                }
            }
        }
    }
    return output_vps;
}

std::vector<Viewpoint> PathPlanner::vp_sample(const Eigen::Vector3d& origin, const std::vector<Eigen::Vector3d>& directions, std::vector<double> dists, int vertex_id) {
    std::vector<Viewpoint> viewpoints;
    // for (const auto& u : directions) {
    for (int i=0; i<(int)directions.size(); ++i) {
        const auto& u = directions[i];
        Viewpoint vp;
        vp.position = origin + directions[i] * (dists[i] + disp_dist);
        double yaw = std::atan2(-u.y(), -u.x());
        vp.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
        vp.corresp_vertex_id = vertex_id;
        viewpoints.push_back(vp);
    }
    return viewpoints;
}

bool PathPlanner::viewpoint_check(const Viewpoint& vp, pcl::KdTreeFLANN<pcl::PointXYZ>& voxel_tree) {
    std::vector<int> ids;
    std::vector<float> dsq;
    pcl::PointXYZ query_point(vp.position.x(), vp.position.y(), vp.position.z());
    if (voxel_tree.radiusSearch(query_point, safe_dist, ids, dsq) > 0) {
        return false;
    }
    return true;
}

bool PathPlanner::viewpoint_similarity(const Viewpoint& a, const Viewpoint& b) {
    return ((a.position - b.position).norm() < viewpoint_merge_dist);
}

void PathPlanner::score_viewpoint(Viewpoint *vp) {
    if (GS.global_pts->empty()) {
        RCLCPP_WARN(node_->get_logger(), "Global Points is empty!");
        vp->score = 0.0;
        vp->covered_voxels.clear();
        return;
    }

    // Track what this VP touched
    int new_count   = 0;
    int total_count = 0;
    vp->covered_voxels.clear();

    // Ray‐cast from VP towards each voxel‐center in global_pts
    for (const auto &pt : GS.global_pts->points) {
        Eigen::Vector3d target(pt.x, pt.y, pt.z);
        Eigen::Vector3d vec = target - vp->position;

        double dist = vec.norm();
        if (dist < safe_dist || dist > max_view_dist) {
            continue;
        }

        Eigen::Vector3d dir = vec / dist;
        // Eigen::Vector3d cam_dir = (vp->orientation * Eigen::Vector3d::UnitX()).normalized();
        Eigen::Vector3d cam_dir = vp->orientation.inverse() * vec;
        const double theta_h = std::atan2(cam_dir.y(), cam_dir.x()); // left/right
        const double theta_v = std::atan2(cam_dir.z(), cam_dir.x()); // up/down

        if (std::abs(theta_h) > (fov_h * 0.5 * M_PI / 180.0)) continue;
        if (std::abs(theta_v) > (fov_v * 0.5 * M_PI / 180.0)) continue;

        
        // Target voxel index
        VoxelIndex tgt_idx {
            int(std::floor(target.x() / voxel_size)),
            int(std::floor(target.y() / voxel_size)),
            int(std::floor(target.z() / voxel_size))
        };

        // sample along the ray to find *first* occupied voxel
        bool occ = false;        
        int info_steps = int(2 * dist / (voxel_size * 0.5));

        for (int i = 1; i <= info_steps; ++i) {
            Eigen::Vector3d sample = vp->position + dir * (dist * i / double(info_steps));
            VoxelIndex sample_idx {
                int(std::floor(sample.x() / voxel_size)),
                int(std::floor(sample.y() / voxel_size)),
                int(std::floor(sample.z() / voxel_size))
            };

            if (sample_idx == tgt_idx) break; // arrived at target...

            // If occluded - break
            if (GS.voxels.count(sample_idx) > 0) {
                occ = true;
                break;
            }
        }

        if (!occ && GS.voxels.count(tgt_idx) > 0) {
            ++total_count;
            vp->covered_voxels.push_back(tgt_idx); // Log seen voxels by this viewpoint
            if (GS.global_seen_voxels.count(tgt_idx) == 0) {
                ++new_count;
            }
        }
    }

    // ratio of new to seen
    vp->score = (total_count > 0)
              ? double(new_count) / double(total_count)
              : 0.0;
}

bool PathPlanner::corridor_obstructed(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2) {
    const double step = voxel_size / 2.0;
    Eigen::Vector3d dir = (p2 - p1).normalized();
    double length = (p2 - p1).norm();

    int n_steps = static_cast<int>(std::ceil(length / step));

    pcl::PointXYZ query;

    std::vector<int> nn_ids;
    std::vector<float> nn_sq_dists;

    for (int i=1 ; i<=n_steps; ++i) {
        Eigen::Vector3d sample = p1 + dir * (i * step);
        query.x = sample.x();
        query.y = sample.y();
        query.z = sample.z();

        int found = global_pts_kdtree.radiusSearch(query, corridor_radius, nn_ids, nn_sq_dists, 1);

        if (found > 0) {
            return true;
        }
    }
    return false;
}

double PathPlanner::distance_to_free_space(const Eigen::Vector3d &p, Eigen::Vector3d& dir) {
    const double max_dist = 2;
    const double step = voxel_size / 2.0;
    for (double d = step; d<=max_dist; d+=step) {
        Eigen::Vector3d sample = p + dir * d;
        VoxelIndex idx = {
            int(std::floor(sample.x() / voxel_size)),
            int(std::floor(sample.y() / voxel_size)),
            int(std::floor(sample.z() / voxel_size))
        };

        if (GS.voxels.count(idx) == 0) {
            return d;
        }
    }
    return max_dist;
}

/* Voxel Grid Map */
void PathPlanner::global_cloud_handler() {
    // Maybe upgrade from hash-map to octree structure??
    for (const auto &pt : local_pts->points) {
        if (pt.z < gnd_th) continue;
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
        VoxelIndex idx = {
            static_cast<int>(std::floor(pt.x / voxel_size)),
            static_cast<int>(std::floor(pt.y / voxel_size)),
            static_cast<int>(std::floor(pt.z / voxel_size))
        };

        GS.voxel_point_count[idx]++;

        if (GS.voxel_point_count[idx] >= 3) {
            GS.voxels.insert(idx);
        }
    }

    GS.global_pts->clear();
    for (const auto &v : GS.voxels) {
        GS.global_pts->points.emplace_back(
            (v.x + 0.5f) * voxel_size,
            (v.y + 0.5f) * voxel_size,
            (v.z + 0.5f) * voxel_size
        );
    }

    if (!GS.global_pts->empty()) global_pts_kdtree.setInputCloud(GS.global_pts); // Set kdtree for point occupancy query
}

void PathPlanner::update_seen_cloud(Viewpoint *vp) {
    if (!GS.global_seen_cloud) {
        GS.global_seen_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    };

    score_viewpoint(vp); // Update actual seen voxels (from the previous estimate)

    for (const auto& idx : vp->covered_voxels) {
        if (GS.global_seen_voxels.insert(idx).second) {
            GS.global_seen_cloud->points.emplace_back(
                (idx.x + 0.5f) * voxel_size,
                (idx.y + 0.5f) * voxel_size,
                (idx.z + 0.5f) * voxel_size);
        }
    }
}

