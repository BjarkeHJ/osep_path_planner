#ifndef PLANNER_HPP_
#define PLANNER_HPP_

#include <iostream>
#include <chrono>
#include <pcl/common/common.h>
#include <Eigen/Core>

#define RUN_STEP(fn) \
    do { \
        bool ok = (fn)(); \
        running = ok; \
        if (!ok) return false; \
    } while (0)


struct PlannerConfig {
    int test;
};

struct Viewpoint {
    Eigen::Vector3f position;
    Eigen::Quaternionf orientation;
    int corresp_vid;

    float score = 0.0f;
    bool in_path = false;
    bool visited = false;
};

struct PlannerData {
    Viewpoint start;
    Viewpoint end;

    std::vector<Viewpoint> all_vpts;
    
};


class PathPlanner {
public:
    PathPlanner(const PlannerConfig& cfg);
    bool planner_run();


private:
    /* Functions */
    bool generate_path();

    /* Helper */


    /* Params */
    PlannerConfig cfg_;
    bool running;

    /* Data */
    PlannerData PD;
};


#endif