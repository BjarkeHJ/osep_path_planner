/* 

Main skeleton extraction algorithm 
Implementation of ROSA Point Extraction 
with Implementation of Fallback strategies 
for Planar Surfaces

*/

#include "rosa_main.hpp"

SkelEx::SkelEx(rclcpp::Node::SharedPtr node) : node_(node) 
{
}

void SkelEx::init() {
    RCLCPP_INFO(node_->get_logger(), "Initializing Module: Local Skeleton Extractor");
    /* Params */
    // Stuff from launch file (Todo)...

    /* Data */
    SS.pts_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    SS.normals_.reset(new pcl::PointCloud<pcl::Normal>);
    SS.vertices_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    pset_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
}

void SkelEx::main() {
    auto t_start = std::chrono::high_resolution_clock::now();

    distance_filter();
    pcd_size_ = SS.pts_->points.size();
    
    int pts_lim = static_cast<int>(std::floor(max_points*0.8));
    if (pcd_size_ < pts_lim) {
        RCLCPP_INFO(node_->get_logger(), "Not enough points to safely compute... Size: %d / %d", pcd_size_, pts_lim);
        return;
    }
    RCLCPP_INFO(node_->get_logger(), "Point Cloud Size: %d", pcd_size_);

    normal_estimation();
    RCLCPP_INFO(node_->get_logger(), "Downsampled Point Cloud Size: %d", pcd_size_);

    similarity_neighbor_extraction();
    drosa();
    dcrosa();
    vertex_sampling();
    vertex_smooth();
    vertex_recenter();
    get_vertices();

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_elapsed = t_end - t_start;
    RCLCPP_INFO(node_->get_logger(), "Time Elapsed: %f seconds", t_elapsed.count());
}

void SkelEx::distance_filter() {
    /* Distance Filtering */
    pcl::PassThrough<pcl::PointXYZ> ptf;
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    ptf.setInputCloud(SS.pts_);
    ptf.setFilterFieldName("x");
    ptf.setFilterLimits(-pts_dist_lim, pts_dist_lim);
    ptf.filter(*temp_cloud);  
    
    ptf.setInputCloud(temp_cloud);
    ptf.setFilterFieldName("y");
    ptf.setFilterLimits(-pts_dist_lim, pts_dist_lim);
    ptf.filter(*SS.pts_);
}

void SkelEx::normal_estimation() {
    // Normalization
    // pcl::PointXYZ min, max;
    // pcl::getMinMax3D(*SS.pts_, min, max);
    // double x_scale, y_scale, z_scale;
    // x_scale = max.x - min.x;
    // y_scale = max.y - min.y;
    // z_scale = max.z - min.z;
    // norm_scale = std::max(x_scale, std::max(y_scale, z_scale));
    // pcl::compute3DCentroid(*SS.pts_, centroid);

    // for (int i=0; i<pcd_size_; ++i) {
    //     SS.pts_->points[i].x = (SS.pts_->points[i].x - centroid(0)) / norm_scale;
    //     SS.pts_->points[i].y = (SS.pts_->points[i].y - centroid(1)) / norm_scale;
    //     SS.pts_->points[i].z = (SS.pts_->points[i].z - centroid(2)) / norm_scale;
    // }

    // Surface Normal Estimation
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr ne_tree(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setInputCloud(SS.pts_);
    ne.setSearchMethod(ne_tree);
    ne.setKSearch(ne_KNN);
    ne.compute(*SS.normals_);

    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_w_normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*SS.pts_, *SS.normals_, *cloud_w_normals);

    // VoxelGrid Downsampling
    pcl::VoxelGrid<pcl::PointNormal> vgf;
    leaf_size_ds = 0.3;
    // leaf_size_ds = 0.001;
    int iter=0;
    while (pcd_size_ > max_points) {
        vgf.setInputCloud(cloud_w_normals);
        vgf.setLeafSize(leaf_size_ds,leaf_size_ds,leaf_size_ds);
        vgf.filter(*cloud_w_normals);
        pcd_size_ = cloud_w_normals->points.size();
        if (pcd_size_ <= max_points) break;
        leaf_size_ds += 0.05;
        // leaf_size_ds += 0.005;
        iter++;
    }
    RCLCPP_INFO(node_->get_logger(), "VGF Iterations: %d", iter);
    
    pcl::KdTreeFLANN<pcl::PointNormal> kdtree;
    kdtree.setInputCloud(cloud_w_normals);

    std::vector<int> indices;
    std::vector<float> sqr_distances;
    pcl::PointCloud<pcl::PointNormal>::Ptr filtered_points(new pcl::PointCloud<pcl::PointNormal>);

    double density_r = 3 * leaf_size_ds;
    int min_nbs = 8;
    for (size_t i = 0; i < cloud_w_normals->size(); ++i) {
        int neighbors = kdtree.radiusSearch(cloud_w_normals->points[i], density_r, indices, sqr_distances);
        if (neighbors >= min_nbs) {
            filtered_points->points.push_back(cloud_w_normals->points[i]);
        }
    }
    
    *cloud_w_normals = *filtered_points;
    pcd_size_ = cloud_w_normals->points.size();
    
    SS.pts_->clear();
    SS.normals_->clear();
    SS.pts_matrix.resize(pcd_size_, 3);
    SS.nrs_matrix.resize(pcd_size_, 3);
    pcl::PointXYZ pt;
    pcl::Normal nrm;

    for (int i=0; i<pcd_size_; ++i) {
        pt.x = cloud_w_normals->points[i].x;
        pt.y = cloud_w_normals->points[i].y;
        pt.z = cloud_w_normals->points[i].z;

        Eigen::Vector3d pt_vec(pt.x, pt.y, pt.z);
        Eigen::Vector3d nrm_vec(cloud_w_normals->points[i].normal_x,
                                cloud_w_normals->points[i].normal_y,
                                cloud_w_normals->points[i].normal_z);
        // Eigen::Vector3d lidar_pos = (-centroid.head<3>().transpose()) / norm_scale;
        Eigen::Vector3d lidar_pos(0.0, 0.0, 0.0);
        Eigen::Vector3d to_sensor = lidar_pos - pt_vec;

        if (nrm_vec.dot(to_sensor) < 0) {
            nrm_vec = -nrm_vec;
        }

        nrm.normal_x = nrm_vec(0);
        nrm.normal_y = nrm_vec(1);
        nrm.normal_z = nrm_vec(2);
        SS.pts_->points.push_back(pt);
        SS.normals_->points.push_back(nrm);
        SS.pts_matrix.row(i) = pt_vec;
        SS.nrs_matrix.row(i) = nrm_vec;
    }
}

void SkelEx::similarity_neighbor_extraction() {
    SS.neighs.clear();
    SS.neighs.resize(pcd_size_);

    pcl::KdTreeFLANN<pcl::PointXYZ> sim_tree;
    sim_tree.setInputCloud(SS.pts_);
    pcl::PointXYZ search_pt, p1, p2;
    pcl::Normal v1, v2;
    std::vector<int> indxs;
    std::vector<float> sq_dists;
    double w1, w2, w;
    double radius_r = 10*leaf_size_ds;
    double th_sim = 0.1*leaf_size_ds;

    for (int i=0; i<pcd_size_; ++i) {
        std::vector<int>().swap(indxs);
        std::vector<float>().swap(sq_dists);
        p1 = SS.pts_->points[i];
        v1 = SS.normals_->points[i];
        sim_tree.radiusSearch(p1, radius_r, indxs, sq_dists);
        std::vector<int> temp_neighs;

        for (int j=0; j<(int)indxs.size(); ++j) {
            p2 = SS.pts_->points[indxs[j]];
            v2 = SS.normals_->points[indxs[j]];
            w1 = similarity_metric(p1, v1, p2, v2, radius_r);
            w2 = similarity_metric(p2, v2, p1, v1, radius_r);
            w = std::min(w1, w2);
            if (w > th_sim) {
                temp_neighs.push_back(indxs[j]);
            }
        }
    SS.neighs[i] = temp_neighs;
    }
}

void SkelEx::drosa() {
    ExtractTools et_d;
    rosa_init(SS.pts_, SS.normals_);

    SS.surf_neighs.clear();
    SS.surf_neighs.reserve(pcd_size_);
    std::vector<int> temp_surf(k_KNN);
    std::vector<float> nn_sq_dists(k_KNN);
    pcl::KdTreeFLANN<pcl::PointXYZ> surf_tree;
    surf_tree.setInputCloud(SS.pts_);
    for (int i=0; i<pcd_size_; ++i) {
        pcl::PointXYZ search_pt_surf = SS.pts_->points[i];
        surf_tree.nearestKSearch(search_pt_surf, k_KNN, temp_surf, nn_sq_dists);
        SS.surf_neighs.emplace_back(temp_surf);
    }

    /* ROSA Point Orientaiton */
    Eigen::Vector3d var_p, var_v, new_v;
    Eigen::MatrixXd indxs, extract_normals;
    delta = leaf_size_ds;
    for (int n=0; n<drosa_iter; ++n) {
        Eigen::MatrixXd vnew = Eigen::MatrixXd::Zero(pcd_size_, 3);

        for (int pidx=0; pidx<pcd_size_; ++pidx) {
            var_p = pset.row(pidx);
            var_v = vset.row(pidx);
            indxs = compute_active_samples(pidx, var_p, var_v);
            extract_normals = et_d.rows_ext_M(indxs, SS.nrs_matrix);

            if (extract_normals.rows() > 0) {
                new_v = compute_symmetrynormal(extract_normals);
                vnew.row(pidx) = new_v.transpose();
                vvar(pidx, 0) = symmnormal_variance(new_v, extract_normals);
            }
            else {
                vvar(pidx, 0) = 0.0;
            }
        }

        vset = vnew; // For next iteration

        Eigen::MatrixXd offset(vvar.rows(), vvar.cols());
        offset.setOnes();
        offset = 1e-5*offset; //Avoid division by zero

        // Weighting the variance values to suppress large variance and emphasize low-variance regions
        vvar = (vvar.cwiseAbs2().cwiseAbs2() + offset).cwiseInverse(); // 1/(vvar⁴)

        /* Smoothing */
        for (int p=0; p<pcd_size_; ++p) {
            std::vector<int> &surf = SS.surf_neighs[p];
            if (surf.empty()) continue;
            
            Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, 1>> snidxs_map(surf.data(), surf.size());
            Eigen::MatrixXd snidxs_d = snidxs_map.cast<double>();    
            Eigen::MatrixXd vset_ex = et_d.rows_ext_M(snidxs_d, vset);
            Eigen::MatrixXd vvar_ex = et_d.rows_ext_M(snidxs_d, vvar);
            vset.row(p) = symmnormal_smooth(vset_ex, vvar_ex);
        }
    }

    /* ROSA Point Position */
    std::vector<int> poorIdx; 
    pcl::PointCloud<pcl::PointXYZ>::Ptr goodPts(new pcl::PointCloud<pcl::PointXYZ>);
    std::map<Eigen::Vector3d, Eigen::Vector3d, Vector3dCompare> goodPtsPset;
    double PLANAR_TH = 0.1;
    for (int pIdx=0; pIdx<pcd_size_; ++pIdx) {
        Eigen::Vector3d var_p_p = pset.row(pIdx);
        Eigen::Vector3d var_v_p = vset.row(pIdx);
        Eigen::MatrixXd indxs_p = compute_active_samples(pIdx, var_p_p, var_v_p); // Extract plane slice indices
        Eigen::MatrixXd extract_pts = et_d.rows_ext_M(indxs_p, SS.pts_matrix);
        Eigen::MatrixXd extract_nrs = et_d.rows_ext_M(indxs_p, SS.nrs_matrix);

        // Based on the plane slice normals - detect planar surfaces
        Eigen::Vector3d evals = PCA(extract_nrs);
        double planar_score = (evals(1) - evals(0)) / evals(2);
        Eigen::Vector3d center;
        if (planar_score < PLANAR_TH) {
            Eigen::RowVector3d slice_pt_mean = extract_pts.colwise().mean();
            Eigen::RowVector3d slice_nrm_mean = extract_nrs.colwise().mean();
            center = slice_pt_mean - slice_nrm_mean * (max_projection_range / 2); // Project skeleton based on mean point and mean normal
        }
        else {
            center = closest_projection_point(extract_pts, extract_nrs);
        }
        
        bool valid_center = (center - var_p_p).norm() < max_projection_range && center.maxCoeff() < 1e7;
        if (valid_center) {
            pset.row(pIdx) = center;
            const pcl::PointXYZ& src_pt = SS.pts_->points[pIdx];
            goodPts->push_back(src_pt);
            goodPtsPset[{src_pt.x, src_pt.y, src_pt.z}] = center;
        } else {
            poorIdx.push_back(pIdx);
        }
    }

    if (goodPts->points.empty()) {
        RCLCPP_INFO(node_->get_logger(), "WARNING: goodPts Cloud EMPTY!");
        return;
    }

    /* Reposition poor points to nearest good point */
    pcl::KdTreeFLANN<pcl::PointXYZ> rosa_tree;
    rosa_tree.setInputCloud(goodPts);

    std::vector<int> pair_id(1);
    std::vector<float> nn_sq_dist(1);
    for (int poor_id : poorIdx) {
        pcl::PointXYZ &search_pt = SS.pts_->points[poor_id];
        if (rosa_tree.nearestKSearch(search_pt, 1, pair_id, nn_sq_dist) > 0) {
            pcl::PointXYZ &nearest_pt = goodPts->points[pair_id[0]];
            Eigen::Vector3d query_key{nearest_pt.x, nearest_pt.y, nearest_pt.z};

            auto it = goodPtsPset.find(query_key);
            if (it != goodPtsPset.end()) {
                pset.row(poor_id) = it->second;
            }
            else {
                RCLCPP_WARN(node_->get_logger(), "Projection fallback: nearest good point not found in map.");
            }
        }
        else {
            RCLCPP_WARN(node_->get_logger(), "Projection fallback: no neighbor found for poor point index %d", poor_id);
        }
    }
}

void SkelEx::dcrosa() {
    ExtractTools et_dc;
    Eigen::MatrixXd newpset(pcd_size_, 3);
    double CONFIDENCE_TH = 0.5;

    for (int n=0; n<dcrosa_iter; ++n) {
        for (int i=0; i<pcd_size_; ++i) {
            auto& neigh = SS.neighs[i];
            if (!neigh.empty()) {
                Eigen::MatrixXi int_indxs = Eigen::Map<Eigen::MatrixXi>(SS.neighs[i].data(), SS.neighs[i].size(), 1);
                Eigen::MatrixXd indxs = int_indxs.cast<double>();
                Eigen::MatrixXd neigh_pts = et_dc.rows_ext_M(indxs, pset);
                newpset.row(i) = neigh_pts.colwise().mean();
            }
            else {
                newpset.row(i) = pset.row(i);
            }
        }
        pset = newpset;

        pset_cloud->clear();
        pset_cloud->width = pset.rows();
        pset_cloud->height = 1;
        pset_cloud->points.resize(pcd_size_);
        for (int i=0; i<pcd_size_; ++i) {
            pset_cloud->points[i].x = pset(i,0);
            pset_cloud->points[i].y = pset(i,1);
            pset_cloud->points[i].z = pset(i,2);
        }
        
        pcl::KdTreeFLANN<pcl::PointXYZ> pset_tree;
        pset_tree.setInputCloud(pset_cloud);
        Eigen::VectorXd conf = Eigen::VectorXd::Zero(pcd_size_);
        newpset = pset;

        for (int i=0; i<pcd_size_; ++i) {
            std::vector<int> pt_idx(k_KNN);
            std::vector<float> pt_dists(k_KNN);
            if (pset_tree.nearestKSearch(pset_cloud->points[i], k_KNN, pt_idx, pt_dists) > 0) {
                Eigen::MatrixXd neighs_pts(k_KNN, 3);
                for (int j=0; j<k_KNN; ++j) {
                    neighs_pts.row(j) = pset.row(pt_idx[j]);
                }
                Eigen::Vector3d local_mean = neighs_pts.colwise().mean();
                neighs_pts.rowwise() -= local_mean.transpose();

                Eigen::BDCSVD<Eigen::MatrixXd> svd(neighs_pts, Eigen::ComputeThinU | Eigen::ComputeThinV);
                conf(i) = svd.singularValues()(0) / (svd.singularValues().sum());

                if (conf(i) > CONFIDENCE_TH) {
                    newpset.row(i) = svd.matrixU().col(0).transpose() * (svd.matrixU().col(0) * (pset.row(i) - local_mean.transpose())) + local_mean.transpose();
                }
            }
        }
        pset = newpset;
    }
}

void SkelEx::vertex_sampling() {
    ExtractTools et_vs;
    pcl::PointXYZ pset_pt;
    pset_cloud->clear();
    for (int i=0; i<pcd_size_; i++) {
        pset_pt.x = pset(i,0);
        pset_pt.y = pset(i,1);
        pset_pt.z = pset(i,2);
        pset_cloud->points.push_back(pset_pt);
    }

    // mindst stores the minimum squared distance from each unassigned point to the nearest assigned skeleton point. 
    Eigen::MatrixXd mindst = Eigen::MatrixXd::Constant(pcd_size_, 1, std::numeric_limits<double>::quiet_NaN()); 
    SS.corresp = Eigen::MatrixXd::Constant(pcd_size_, 1, -1); // initialized with value -1

    Eigen::MatrixXi int_nidxs;
    Eigen::MatrixXd nIdxs;
    Eigen::MatrixXd extract_corresp;
    pcl::PointXYZ search_point;
    std::vector<int> indxs;
    std::vector<float> radius_squared_distance;
    
    pcl::KdTreeFLANN<pcl::PointXYZ> fps_tree;
    fps_tree.setInputCloud(pset_cloud);
    SS.skelver.resize(0, 3);
    
    double sample_radius = 1.0 * leaf_size_ds;

    // Farthest Point Sampling (FPS) / Skeletonization / Vertex selection
    for (int k=0; k<pcd_size_; k++) {
        if (SS.corresp(k,0) != -1) continue; // skip already assigned points - Will only proceed if gaps larger than search radius in ROSA points (after 1st iter)
        mindst(k,0) = 1e8; // set large to ensure update

        // run while ANY element in corresp is still -1
        while (!((SS.corresp.array() != -1).all())) {
            int maxIdx = argmax_eigen(mindst); // maxIdx represents the most distant unassigned point

            // If the largest distance value is zero... I.e. all remaining unassinged points are with the radius
            if (!std::isnan(mindst(maxIdx, 0)) && mindst(maxIdx,0) == 0) break;

            // The current search point
            search_point.x = pset(maxIdx,0);
            search_point.y = pset(maxIdx,1);
            search_point.z = pset(maxIdx,2);
            
            // Search for points within the sample_radius of the current search point. 
            // The indices of the nearest points are set in indxs
            indxs.clear();
            radius_squared_distance.clear();
            fps_tree.radiusSearch(search_point, sample_radius, indxs, radius_squared_distance);

            int_nidxs = Eigen::Map<Eigen::MatrixXi>(indxs.data(), indxs.size(), 1); // structures the column vector of the nearest neighbours 
            nIdxs = int_nidxs.cast<double>();
            extract_corresp = et_vs.rows_ext_M(nIdxs, SS.corresp); // Extract the section corresp according to the indices of the nearest points

            // If all neighbours wihtin sample_radius already has been assigned (neq to -1) the current point is not needed as vertex
            if ((extract_corresp.array() != -1).all()) {
                mindst(maxIdx,0) = 0;
                continue; // Go to loop start
            }

            // If all neighbours had not been assigned to a corresponding vertex, the current search point is chosen as a new vertex.
            SS.skelver.conservativeResize(SS.skelver.rows()+1, SS.skelver.cols()); // adds one vertex
            SS.skelver.row(SS.skelver.rows()-1) = pset.row(maxIdx);

            // for every point withing the sample_radius
            for (int z=0; z<(int)indxs.size(); z++) {

                // if the distance value at this index is unassigned OR if a previous assignment has a larger distance
                // the point is assigned to the new vertex
                // this ensures that every point is assigned to their closest vertex
                if (std::isnan(mindst(indxs[z],0)) || mindst(indxs[z],0) > radius_squared_distance[z]) {
                    mindst(indxs[z],0) = radius_squared_distance[z]; // update minimum distance to closest vertex
                    SS.corresp(indxs[z], 0) = SS.skelver.rows() - 1; // Keeps track of which skeleton vertice each point corresponds to (0, 1, 2, 3...)
                }
            }
        }
    }
}

void SkelEx::vertex_recenter() {
    ExtractTools et_vr;
    std::vector<std::vector<int>> vertex_to_pts(SS.skelver.rows());

    // Inverse correspondence map
    for (int i=0; i<SS.corresp.rows(); ++i) {
        int vert_idx = static_cast<int>(SS.corresp(i,0));
        vertex_to_pts[vert_idx].push_back(i);
    }

    for (int i=0; i<SS.skelver.rows(); ++i) {
        auto &idxs = vertex_to_pts[i];

        Eigen::MatrixXi c_indxs = Eigen::Map<Eigen::MatrixXi>(idxs.data(), idxs.size(), 1);
        Eigen::MatrixXd c_indxs_d = c_indxs.cast<double>();
        Eigen::MatrixXd extract_pts = et_vr.rows_ext_M(c_indxs_d, SS.pts_matrix);
        Eigen::MatrixXd extract_nrs = et_vr.rows_ext_M(c_indxs_d, SS.nrs_matrix);

        Eigen::Vector3d eucl_center = extract_pts.colwise().mean();
        Eigen::Vector3d current = SS.skelver.row(i);

        Eigen::Vector3d fuse_center = alpha * current + (1.0 - alpha) * eucl_center;
        SS.skelver.row(i) = fuse_center.transpose();
    }
}

void SkelEx::vertex_smooth() {
    const int iterations = 5;
    const double r_smooth = 5.0;
    std::vector<int> indxs;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

    // Build skeleton vertex cloud for KNN
    pcl::PointCloud<pcl::PointXYZ>::Ptr skel_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    skel_cloud->resize(SS.skelver.rows());
    for (int i = 0; i < SS.skelver.rows(); ++i) {
        (*skel_cloud)[i].x = SS.skelver(i, 0);
        (*skel_cloud)[i].y = SS.skelver(i, 1);
        (*skel_cloud)[i].z = SS.skelver(i, 2);
    }
    kdtree.setInputCloud(skel_cloud);

    Eigen::MatrixXd new_skelver = SS.skelver;

    for (int iter = 0; iter < iterations; ++iter) {
        Eigen::MatrixXd current = new_skelver;

        for (int i = 0; i < current.rows(); ++i) {

            std::vector<int> neigh_indices;
            std::vector<float> neigh_dists;
            if (kdtree.radiusSearch((*skel_cloud)[i], r_smooth, neigh_indices, neigh_dists) > 0) {

                Eigen::MatrixXd neigh_pos(neigh_indices.size(), 3);
                for (int j = 0; j < (int)neigh_indices.size(); ++j) {
                    neigh_pos.row(j) = current.row(neigh_indices[j]);
                }

                Eigen::RowVector3d mean = neigh_pos.colwise().mean();
                neigh_pos.rowwise() -= mean;

                // Compute principal direction using SVD
                Eigen::JacobiSVD<Eigen::MatrixXd> svd(neigh_pos, Eigen::ComputeThinU | Eigen::ComputeThinV);
                Eigen::Vector3d principal_dir = svd.matrixV().col(0);  // dominant axis

                Eigen::Vector3d delta = current.row(i).transpose() - mean.transpose();
                Eigen::Vector3d proj = mean.transpose() + principal_dir * (principal_dir.dot(delta));

                new_skelver.row(i) = (1.0 - alpha) * current.row(i) + alpha * proj.transpose();
            }
        }

        // Update the point cloud for the next iteration
        for (int i = 0; i < new_skelver.rows(); ++i) {
            (*skel_cloud)[i].x = new_skelver(i, 0);
            (*skel_cloud)[i].y = new_skelver(i, 1);
            (*skel_cloud)[i].z = new_skelver(i, 2);
        }
    }

    SS.skelver = new_skelver;
}

void SkelEx::get_vertices() {
    // Transform local vertices for global skeleton increment...
    SS.vertices_->clear();
    pcl::PointXYZ pt;
    for (int i=0; i<(int)SS.skelver.rows(); ++i) {
        Eigen::Vector3d ver_tf = SS.skelver.row(i);
        ver_tf = tf_rot * ver_tf + tf_trans;
        if (ver_tf(2) < gnd_th) continue; //skip vertices close to ground...

        pt.x = ver_tf(0);
        pt.y = ver_tf(1);
        pt.z = ver_tf(2);
        SS.vertices_->points.push_back(pt);
    }
}


/* Helper Functions */
double SkelEx::similarity_metric(pcl::PointXYZ &p1, pcl::Normal &v1, pcl::PointXYZ &p2, pcl::Normal &v2, double range_r) {
    double Fs = 5.0;
    double k = 0.0;
    double dist, vec_dot, w;
    Eigen::Vector3d p1_, p2_, v1_, v2_;

    p1_ << p1.x, p1.y, p1.z;
    p2_ << p2.x, p2.y, p2.z;
    v1_ << v1.normal_x, v1.normal_y, v1.normal_z;
    v2_ << v2.normal_x, v2.normal_y, v2.normal_z;

    // the displacement vector + the projection of the displacement vector onto the search point normal vector
    // If the displacement vector is perpendicular with the normal vector (projection = 0) the two points both lie in the plane given by the normal vector
    // If that is the case, the contribution to the distance metric is not increased
    // Else the metric is increased... 
    dist = (p1_ - p2_ + Fs*((p1_ - p2_).dot(v1_))*v1_).norm();
    dist = dist/range_r;

    if (dist <= 1) {
        k = 2*pow(dist, 3) - 3*pow(dist, 2) + 1;
    }

    // Projection of v1 onto v2
    vec_dot = v1_.dot(v2_);
    // max(0, vec_dot) to not include antiparallel normal vectors
    w = k*pow(std::max(0.0, vec_dot), 2);
    return w;
}

void SkelEx::rosa_init(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud<pcl::Normal>::Ptr &normals) {
    Eigen::Matrix3d M;
    Eigen::Vector3d normal_v;
    pset.resize(pcd_size_, 3);
    vset.resize(pcd_size_, 3);
    vvar.resize(pcd_size_, 1);
    for (int i=0; i<pcd_size_; ++i) {
        pset(i,0) = cloud->points[i].x;
        pset(i,1) = cloud->points[i].y;
        pset(i,2) = cloud->points[i].z;
        normal_v(0) = normals->points[i].normal_x;
        normal_v(1) = normals->points[i].normal_y;
        normal_v(2) = normals->points[i].normal_z;
        M = create_orthonormal_frame(normal_v);
        vset.row(i) = M.row(1); // Vector orthogonal to surface normal (Plane tangent vector)
    }
}

Eigen::Matrix3d SkelEx::create_orthonormal_frame(Eigen::Vector3d &v) {
    v = v/v.norm();
    double TH_ZERO = 1e-10;
    Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
    M(0,0) = v(0); 
    M(0,1) = v(1); 
    M(0,2) = v(2);
    Eigen::Vector3d new_vec, temp_vec;

    // The outer for loops finds an orthonormal basis
    for (int i=1; i<3; ++i) {
        new_vec.setRandom();
        new_vec = new_vec/new_vec.norm();
  
        while (abs(1.0 - v.dot(new_vec)) < TH_ZERO) {
          // Run until vector (not too parallel) is found... Avoid colinear vectors
          new_vec.setRandom();
          new_vec = new_vec / new_vec.norm();
        }
 
        // Gramm-Schmidt process to find orthogonal vectors
        for (int j=0; j<i; ++j) {
          temp_vec = (new_vec - new_vec.dot(M.row(j)) * (M.row(j).transpose()));
          new_vec = temp_vec/temp_vec.norm();
        }
  
        M(i,0) = new_vec(0);
        M(i,1) = new_vec(1);
        M(i,2) = new_vec(2);
      }
    return M;
}

Eigen::MatrixXd SkelEx::compute_active_samples(int &idx, Eigen::Vector3d &p_cut, Eigen::Vector3d &v_cut) {
    // Extracts an index-vector masked with the indices on the plane slice
    Eigen::MatrixXd out_indxs(pcd_size_, 1);
    int out_size = 0;
    std::vector<int> isoncut(pcd_size_, 0); // On cut mask

    std::vector<double> p(3); // Current point
    p[0] = p_cut(0);
    p[1] = p_cut(1);
    p[2] = p_cut(2);
    std::vector<double> n(3); // Corresponding plane normal vector
    n[0] = v_cut(0);
    n[1] = v_cut(1);
    n[2] = v_cut(2);

    std::vector<double> Pi(3); // Point to check if isoncut
    for (int pIdx=0; pIdx<pcd_size_; pIdx++) {
        Pi[0] = SS.pts_->points[pIdx].x;
        Pi[1] = SS.pts_->points[pIdx].y;
        Pi[2] = SS.pts_->points[pIdx].z;

        // Determine if the current point is included in the plane slice. That is within delta distance from the plane...
        // Distance is calculated as d = n*(p - P)
        // using the plane equation: https://tutorial.math.lamar.edu/classes/calciii/eqnsofplanes.aspx
        if (fabs(n[0]*(p[0]-Pi[0]) + n[1]*(p[1]-Pi[1]) + n[2]*(p[2]-Pi[2])) < delta) {
            isoncut[pIdx] = 1;
        }
    }

    // Flood-fill algorithm to ensure that the other regions of plane intersection is not included...
    std::vector<int> queue;
    queue.reserve(pcd_size_); // Allocate memory
    queue.emplace_back(idx); // Insert the seed-point for region growing

    int curr;
    while (!queue.empty()) {
        curr = queue.back();
        queue.pop_back();
        isoncut[curr] = 2;
        out_indxs(out_size++, 0) = curr; //Add to final output... 

        // For the current point iterate through its neighs (Normal away neighs)...
        for (size_t i = 0; i < SS.neighs[curr].size(); ++i) {
            // If a nb is on-cut...
            if (isoncut[SS.neighs[curr][i]] == 1) {
                isoncut[SS.neighs[curr][i]] = 3; // Mark as part of the region
                queue.emplace_back(SS.neighs[curr][i]); // Set next search point
            }
        }
    }
    out_indxs.conservativeResize(out_size, 1); // Reduces the size down to an array of indices corresponding to the active samples
    return out_indxs;
}

Eigen::Vector3d SkelEx::compute_symmetrynormal(Eigen::MatrixXd& local_normals) {
    // This function determines the vector that minimizes the variance of the angle between local normals and the vector.
    // This can be interpreted as the "direction" of the skeleton inside the structure...
    // The symmetry normal will be the normal vector of the best fit plane of points corresponding to the local_normals

    Eigen::Matrix3d M; Eigen::Vector3d vec;
    int size = local_normals.rows();
    double Vxx, Vyy, Vzz, Vxy, Vyx, Vxz, Vzx, Vyz, Vzy;

    // Variances: Computing the mean squared value and substracting the mean squared value -> Variance = E[X²] - E[X]²
    Vxx = local_normals.col(0).cwiseAbs2().sum() / size - pow(local_normals.col(0).sum(), 2) / pow(size, 2);
    Vyy = local_normals.col(1).cwiseAbs2().sum() / size - pow(local_normals.col(1).sum(), 2) / pow(size, 2);
    Vzz = local_normals.col(2).cwiseAbs2().sum() / size - pow(local_normals.col(2).sum(), 2) / pow(size, 2);

    // Covariances: Computing the mean of the product of 2 components and subtracting the product of the means of each components -> Covariance = E[XY] - E[X]E[Y]
    Vxy = 2*(local_normals.col(0).cwiseProduct(local_normals.col(1))).sum()/size - 2*local_normals.col(0).sum()*local_normals.col(1).sum()/pow(size, 2);
    Vyx = Vxy;
    Vxz = 2*(local_normals.col(0).cwiseProduct(local_normals.col(2))).sum()/size - 2*local_normals.col(0).sum()*local_normals.col(2).sum()/pow(size, 2);
    Vzx = Vxz;
    Vyz = 2*(local_normals.col(1).cwiseProduct(local_normals.col(2))).sum()/size - 2*local_normals.col(1).sum()*local_normals.col(2).sum()/pow(size, 2);
    Vzy = Vyz;
    M << Vxx, Vxy, Vxz, Vyx, Vyy, Vyz, Vzx, Vzy, Vzz;

    // Perform singular-value-decomposition on the Covariance matrix M = U(Sigma)V^T
    Eigen::BDCSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3d U = svd.matrixU();
    // The last column of the matrix U corresponds to the smallest singular value (in Sigma)
    // This in turn represents the direction of smallest variance
    // I.e. for the plance slice -> plane normal. 
    vec = U.col(M.cols()-1);
    return vec;
}

double SkelEx::symmnormal_variance(Eigen::Vector3d& symm_nor, Eigen::MatrixXd& local_normals) {
    // Computes the variance of the local normal vectors projected onto a symmetric normal vector
    Eigen::VectorXd alpha;
    int num = local_normals.rows();

    // calculate the projection of each local normal on the symmetry normal... 
    alpha = local_normals * symm_nor; // Inner product between the symm_nor and each row (normal) in local_normals

    // Calculate sample variance of the projections
    double var;
    var = alpha.squaredNorm() / num - pow(alpha.mean(), 2); // sum(alphas)/N - mean(alpha)²
    if (num > 1) {
        var /= (num - 1.0); // *1/(N-1)
    }
    return var;
}

Eigen::Vector3d SkelEx::symmnormal_smooth(Eigen::MatrixXd& V, Eigen::MatrixXd& w) {
    // Compute weighted covariance matrix using symmetry normals and variances
    Eigen::Matrix3d M = Eigen::Matrix3d::Zero();

    for (int i = 0; i < V.rows(); ++i) {
        M += w(i, 0) * V.row(i).transpose() * V.row(i);
    }

    Eigen::BDCSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU);
    return svd.matrixU().col(0);
}

Eigen::Vector3d SkelEx::closest_projection_point(const Eigen::MatrixXd& P, const Eigen::MatrixXd& V) {
    const int N = P.rows();
    if (N == 0) return Eigen::Vector3d::Constant(1e8);

    // Precompute reused components
    const auto& Vx = V.col(0);
    const auto& Vy = V.col(1);
    const auto& Vz = V.col(2);
    const auto& Px = P.col(0);
    const auto& Py = P.col(1);
    const auto& Pz = P.col(2);

    const Eigen::ArrayXd Vx2 = Vx.array().square();
    const Eigen::ArrayXd Vy2 = Vy.array().square();
    const Eigen::ArrayXd Vz2 = Vz.array().square();

    // Matrix M (3x3) – symmetric
    Eigen::Matrix3d M;
    M(0,0) = (Vy2 + Vz2).sum();
    M(1,1) = (Vx2 + Vz2).sum();
    M(2,2) = (Vx2 + Vy2).sum();

    M(0,1) = M(1,0) = -(Vx.array() * Vy.array()).sum();
    M(0,2) = M(2,0) = -(Vx.array() * Vz.array()).sum();
    M(1,2) = M(2,1) = -(Vy.array() * Vz.array()).sum();

    // Vector B
    Eigen::Vector3d B;
    B(0) = (Px.array() * (Vy2 + Vz2)).sum()
         - (Vx.array() * Vy.array() * Py.array()).sum()
         - (Vx.array() * Vz.array() * Pz.array()).sum();

    B(1) = (Py.array() * (Vx2 + Vz2)).sum()
         - (Vy.array() * Vx.array() * Px.array()).sum()
         - (Vy.array() * Vz.array() * Pz.array()).sum();

    B(2) = (Pz.array() * (Vx2 + Vy2)).sum()
         - (Vz.array() * Vx.array() * Px.array()).sum()
         - (Vz.array() * Vy.array() * Py.array()).sum();

    // Use LDLT for stability (symmetric positive semi-definite)
    Eigen::Vector3d X;
    Eigen::LDLT<Eigen::Matrix3d> solver(M);
    if (solver.info() != Eigen::Success) {
        X = Eigen::Vector3d::Constant(1e8);  // Degenerate case fallback
    } else {
        X = solver.solve(B);
    }
    return X;
}

Eigen::Vector3d SkelEx::PCA(Eigen::MatrixXd &normals) {
    if (normals.rows() == 0) {
        return Eigen::Vector3d::Zero();
    }

    Eigen::RowVector3d mean = normals.colwise().mean();
    Eigen::MatrixXd centered = normals.rowwise() - mean;
    Eigen::Matrix3d cov = (centered.transpose() * centered) / double(normals.rows());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_solver(cov);
    if (eig_solver.info() != Eigen::Success) {
        RCLCPP_INFO(node_->get_logger(), "[PCA] Eigen Decomposition Failed...");
    }

    Eigen::Vector3d eigenvalues = eig_solver.eigenvalues();
    return eigenvalues.cwiseSqrt();
}

int SkelEx::argmax_eigen(Eigen::MatrixXd &x) {
    Eigen::MatrixXd::Index maxRow, maxCol;
    x.maxCoeff(&maxRow,&maxCol);
    int idx = maxRow;
    return idx;
}
