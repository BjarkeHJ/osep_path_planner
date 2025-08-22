/* 

Main algorithm for the OSEP viewpoint manager

TODO: Incorporate 2d_local_costmap into the viewpoint sampling process.
TODO: Extract branches from the global skeleton (Move function from gskel)
TODO: 

*/

#include "viewpoint_manager.hpp"

ViewpointManager::ViewpointManager(const ViewpointConfig& cfg) : cfg_(cfg) {

    running = 1;
}


bool ViewpointManager::viewpoint_run() {
    VD.gskel_size = VD.global_skel.size();
    RUN_STEP(viewpoint_sampling);
    int count = 0;

    for (auto v : VD.global_skel) {
        if (v.type == 0) count++;
    }
    std::cout << "Number of vertices: " << VD.gskel_size << std::endl;
    std::cout << "Number of invalid vertices: " << count << std::endl;
    return running;
}


bool ViewpointManager::viewpoint_sampling() {

    return 1;
}



/* Helpers */

std::vector<Viewpoint> ViewpointManager::generate_viewpoint(int id) {


}