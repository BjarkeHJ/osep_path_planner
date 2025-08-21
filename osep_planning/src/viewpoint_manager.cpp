/* 

Main algorithm for the OSEP viewpoint manager

*/

#include "viewpoint_manager.hpp"

ViewpointManager::ViewpointManager(const ViewpointConfig& cfg) : cfg_(cfg) {

}


bool ViewpointManager::viewpoint_manager_run() {
    RUN_STEP(viewpoint_sampling);

    return running;
}


bool ViewpointManager::viewpoint_sampling() {

}