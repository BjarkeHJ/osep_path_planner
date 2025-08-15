#ifndef ROSA_HPP_
#define ROSA_HPP_

#include <pcl/common/common.h>

struct RosaConfig {
    int max_points;
    float alpha;
};

struct CloudData {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pts_;
    pcl::PointCloud<pcl::Normal>::Ptr nrms_;

};


class Rosa {
public:
    explicit Rosa(const RosaConfig& cfg);
    

private:
    /* Functions */
    void preprocess();


    /* Params */
    RosaConfig cfg_;

    /* Data */
    CloudData CD;


};

#endif // ROSA_HPP_