#include "extract_tools.hpp"

Eigen::MatrixXd ExtractTools::rows_ext_V(Eigen::VectorXi ind, Eigen::MatrixXd matrix){
	Eigen::MatrixXd zs1(ind.size(), 1);
	zs1 << (ind.head(ind.size())).cast<double>();
	Eigen::MatrixXd final_matrix(zs1.size(), matrix.cols());
	int num = zs1.size();
	for (int k = 0; k < num; k++)
	{
		final_matrix.row(k) = matrix.row(zs1(k, 0));
	}
	return final_matrix;
}

Eigen::MatrixXd ExtractTools::rows_ext_M(Eigen::MatrixXd ind, Eigen::MatrixXd matrix){
	Eigen::MatrixXd final_matrix(ind.size(), matrix.cols());
	int num = ind.size();
	for (int k = 0; k < num; k++)
	{
		final_matrix.row(k) = matrix.row(ind(k,0));
	}
	return final_matrix;
}

Eigen::MatrixXd ExtractTools::cols_ext_V(Eigen::VectorXi ind, Eigen::MatrixXd matrix){
	Eigen::MatrixXd zs1(ind.size(), 1);
	zs1 << (ind.head(ind.size())).cast<double>();
	Eigen::MatrixXd final_matrix(matrix.rows(), zs1.size());
	int num = zs1.size();
	for (int k = 0; k < num; k++)
	{
		final_matrix.col(k) = matrix.col(zs1(k, 0));
	}
	return final_matrix;
}

Eigen::MatrixXd ExtractTools::cols_ext_M(Eigen::MatrixXd ind, Eigen::MatrixXd matrix){
	Eigen::MatrixXd final_matrix(matrix.rows(), ind.size());
	int num = ind.size();
	for (int k = 0; k < num; k++)
	{
		final_matrix.col(k) = matrix.col(ind(k, 0));
	}
	return final_matrix;
}

Eigen::MatrixXd ExtractTools::rows_del_M(Eigen::MatrixXd ind, Eigen::MatrixXd matrix){
	int num = matrix.rows();
	Eigen::VectorXd xl(num);
	for (int i = 0; i < num; i++)
	{
		xl(i) = i;
	}
	for (int i = 0; i < ind.size(); i++)
	{
		xl.coeffRef(ind(i)) = std::numeric_limits<double>::quiet_NaN();
	}
	Eigen::VectorXd out_index(num - ind.size());
	int index(0);
	for (int i = 0; i < num; i++){
		if (std::isnan(xl(i)))
		{
			continue;
		}
		else
		{
			out_index(index) = i;
		}
		index++;
	}
	Eigen::MatrixXd zs1(out_index.size(), 1);
	zs1 << (out_index.head(out_index.size())).cast<double>();
	Eigen::MatrixXd final_matrix(zs1.size(), matrix.cols());
	int num1 = zs1.size();
	for (int k = 0; k < num1; k++)
	{
		final_matrix.row(k) = matrix.row(zs1(k, 0));
	}
	return final_matrix;
}

Eigen::MatrixXd ExtractTools::cols_del_M(Eigen::MatrixXd ind, Eigen::MatrixXd matrix){
	int num = matrix.rows();
	Eigen::VectorXd xl(num);
	for (int i = 0; i < num; i++)
	{
		xl(i) = i;
	}
	for (int i = 0; i < ind.size(); i++)
	{
		xl.coeffRef(ind(i)) = std::numeric_limits<double>::quiet_NaN();
	}
	Eigen::VectorXd out_index(num - ind.size());
	int index(0);
	for (int i = 0; i < num; i++){
		if (std::isnan(xl(i)))
		{
			continue;
		}
		else
		{
			out_index(index) = i;
		}
		index++;
	}
	Eigen::MatrixXd zs1(out_index.size(), 1);
	zs1 << (out_index.head(out_index.size())).cast<double>();
	Eigen::MatrixXd final_matrix(matrix.rows(), zs1.size());
	int num1 = zs1.size();
	for (int k = 0; k < num1; k++)
	{
		final_matrix.col(k) = matrix.col(zs1(k, 0));
	}
	return final_matrix;
}