#include "tools.h"

#include <cmath>
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

VectorXd calculate_rmse(const vector<VectorXd>& estimations,
                        const vector<VectorXd>& ground_truth) {
  VectorXd rmse = VectorXd::Zero(4);
  if (!estimations.size() || estimations.size() != ground_truth.size()) {
    std::cerr << "calculate_rmse(): Invalid parameters" << std::endl;
    return rmse;
  }
  for (size_t i = 0; i < estimations.size(); ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array()*residual.array();
    rmse += residual;
  }
  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();
  return rmse;
}

MatrixXd calculate_jacobian(const VectorXd& x_state) {
  MatrixXd Hj(3, 4);

  double px = x_state[0];
  double py = x_state[1];
  double vx = x_state[2];
  double vy = x_state[3];

  double c1 = px*px+py*py;
  double c2 = std::sqrt(c1);
  double c3 = (c1*c2);

  if (is_nearly_zero(c3)) {
    std::cerr << "calculate_jacobian(): Division by zero" << std::endl;
    Hj.setZero();
    return Hj;
  }

  Hj << px/c2, py/c2, 0, 0,
        -py/c1, px/c1, 0, 0,
        py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;
  return Hj;
}
