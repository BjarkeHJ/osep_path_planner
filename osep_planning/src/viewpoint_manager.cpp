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
    RUN_STEP(viewpoint_sampling);

    return running;
}


bool ViewpointManager::viewpoint_sampling() {
    fetch_updated_vertices();
    if (VD.updated_vertices.size() == 0) return 0; // No updates -> no new/refined viewpoints -> Exit ViewpointManager

    // If type change -> Sample new viewpoints

    // If position change (significantly) -> sample new viewpoints



    return 1;
}



/* Helpers */
void ViewpointManager::fetch_updated_vertices() {
    if (VD.gskel_size == 0) return;
    VD.updated_vertices.clear();
    for (const auto& v : VD.global_skel) {
        if (v.pos_update || v.type_update) {
            VD.updated_vertices.push_back(v.vid);
        }
    }
}


std::vector<Viewpoint> ViewpointManager::generate_viewpoint(int id) {


}