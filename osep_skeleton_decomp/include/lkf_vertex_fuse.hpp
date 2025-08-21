#ifndef KALMAN_VERTEX_FUSION_
#define KALMAN_VERTEX_FUSION_

#include <Eigen/Core>

struct VertexLKF {
    Eigen::Vector3f x;
    Eigen::Matrix3f P;

    void initFrom(const Eigen::Ref<const Eigen::Vector3f>& x0,
                  const Eigen::Ref<const Eigen::Matrix3f>& P0) {
        x = x0;
        P = P0;
    }

    void update(const Eigen::Ref<const Eigen::Vector3f>& z,
                const Eigen::Ref<const Eigen::Matrix3f>& Q,
                const Eigen::Ref<const Eigen::Matrix3f>& R) {
        // Prediction
        const Eigen::Vector3f x_pred = x;
        const Eigen::Matrix3f P_pred = P + Q;
        // Innovation and Gain
        const Eigen::Matrix3f S = P_pred + R;
        const Eigen::Matrix3f K = P_pred * S.inverse();
        // Correction
        x = x_pred + K * (z - x_pred);
        P = (Eigen::Matrix3f::Identity() - K) * P_pred;
    }
};


// class VertexLKF {
// public: 
//     VertexLKF(float process_noise, float measurement_noise) {
//         Q = Eigen::Matrix3f::Identity() * process_noise;
//         R = Eigen::Matrix3f::Identity() * measurement_noise;
//     }

//     void initialize(Eigen::Vector3f init_pos, Eigen::Matrix3f init_cov) {
//         x = init_pos;
//         P = init_cov;
//     }

//     void update(Eigen::Vector3f &z) {
//         // Prediction
//         Eigen::Vector3f x_pred = x;
//         Eigen::Matrix3f P_pred = P + Q;
//         // Kalman Gain
//         Eigen::Matrix3f K = P_pred * (P_pred + R).inverse();
//         // Correction
//         x = x_pred + K * (z - x_pred);
//         P = (Eigen::Matrix3f::Identity() - K) * P_pred;
//     }

//     Eigen::Vector3f getState() { return x; }
//     Eigen::Matrix3f getCovariance() { return P; }

// private:
//     Eigen::Vector3f x; // State vector
//     Eigen::Matrix3f P; // Covariance vector
//     Eigen::Matrix3f Q; // Process noise covariance
//     Eigen::Matrix3f R; // Measurement noise covariance
// };


#endif