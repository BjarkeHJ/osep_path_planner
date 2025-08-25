/* 

Main paht planning algorithm

*/

#include "planner.hpp"

PathPlanner::PathPlanner(const PlannerConfig& cfg) : cfg_(cfg) {
    running = 1;
}

bool PathPlanner::planner_run() {
    RUN_STEP(generate_path);

    return running;
}

bool PathPlanner::generate_path() {
    if (PD.all_vpts.empty()) return 0; // no viewpoints to plan from
    

    return 1;
}