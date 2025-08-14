#ifndef EXTRACT_TOOLS_
#define EXTRACT_TOOLS_

#include<iostream>
#include<algorithm>
#include<Eigen/Eigen>

class ExtractTools {
public:
	Eigen::MatrixXd rows_ext_V(Eigen::VectorXi ind, Eigen::MatrixXd matrix);
	Eigen::MatrixXd rows_ext_M(Eigen::MatrixXd ind, Eigen::MatrixXd matrix);
	Eigen::MatrixXd cols_ext_V(Eigen::VectorXi ind, Eigen::MatrixXd matrix);
	Eigen::MatrixXd cols_ext_M(Eigen::MatrixXd ind, Eigen::MatrixXd matrix);
	Eigen::MatrixXd rows_del_M(Eigen::MatrixXd ind, Eigen::MatrixXd matrix);
	Eigen::MatrixXd cols_del_M(Eigen::MatrixXd ind, Eigen::MatrixXd matrix);
};

#endif //EXTRACT_TOOLS_