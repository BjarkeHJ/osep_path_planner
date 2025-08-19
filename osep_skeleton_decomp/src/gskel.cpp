/* 

Main algorithm for global incremental skeletonization
a topologically preserving representation of the structure

*/

#include <gskel.hpp>

GSkel::GSkel(const GSkelConfig& cfg) : cfg_(cfg) {
    /* Constructor - Init data structures etc... */
    GD.new_cands.reset(new pcl::PointCloud<pcl::PointXYZ>);
    GD.global_vers_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    

    running = 1;
}

bool GSkel::run_gskel() {
    auto ts = std::chrono::high_resolution_clock::now();

    RUN_STEP(increment_skeleton);

    auto te = std::chrono::high_resolution_clock::now();
    auto telaps = std::chrono::duration_cast<std::chrono::milliseconds>(te-ts).count();
    std::cout << "[GSKEL] Time Elapsed: " << telaps << " ms" << std::endl;
    
    return 1;
}

bool GSkel::increment_skeleton() {
    if (GD.new_cands->points.empty()) return;
    
    return 1;
}



