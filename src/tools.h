#pragma once

#include <cmath>
#include <limits>
#include <vector>

#include "Eigen/Dense"

inline bool is_nearly_zero(double x) {
  return std::fabs(x) < 0.0000001;
}

Eigen::VectorXd calculate_rmse(const std::vector<Eigen::VectorXd>& estimations,
                               const std::vector<Eigen::VectorXd>& ground_truth);

Eigen::MatrixXd calculate_jacobian(const Eigen::VectorXd& x_state);
