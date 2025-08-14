#ifndef KALMAN_VERTEX_FUSION_
#define KALMAN_VERTEX_FUSION_

#include <Eigen/Core>

// struct SkeletonVertex {
//     Eigen::Vector3d position;
//     Eigen::Matrix3d covariance;
//     int obs_count = 0;
//     int unconfirmed_check = 0;
//     bool conf_check = false;
//     bool frozen = false;
// };

class VertexLKF {
public: 
    VertexLKF(double process_noise, double measurement_noise) {
        Q = Eigen::Matrix3d::Identity() * process_noise;
        R = Eigen::Matrix3d::Identity() * measurement_noise;
    }

    void initialize(Eigen::Vector3d init_pos, Eigen::Matrix3d init_cov) {
        x = init_pos;
        P = init_cov;
    }

    void update(Eigen::Vector3d &z) {
        // Prediction
        Eigen::Vector3d x_pred = x;
        Eigen::Matrix3d P_pred = P + Q;
        // Kalman Gain
        Eigen::Matrix3d K = P_pred * (P_pred + R).inverse();
        // Correction
        x = x_pred + K * (z - x_pred);
        P = (Eigen::Matrix3d::Identity() - K) * P_pred;
    }

    Eigen::Vector3d getState() {return x;}
    Eigen::Matrix3d getCovariance() {return P;}

private:
    Eigen::Vector3d x; // State vector
    Eigen::Matrix3d P; // Covariance vector
    Eigen::Matrix3d Q; // Process noise covariance
    Eigen::Matrix3d R; // Measurement noise covariance
};


#endif