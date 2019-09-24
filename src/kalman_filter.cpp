#include "kalman_filter.h"

#include <cmath>
#include <iostream>

#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

void KalmanFilter::init(const VectorXd& x, const MatrixXd& P) {
  x_ = x;
  P_ = P;
  I_ = MatrixXd::Identity(x_.size(), x_.size());
}

void KalmanFilter::predict(const MatrixXd& F, const MatrixXd& Q) {
  x_ = F * x_;
  P_ = F * P_ * F.transpose() + Q;
}

void KalmanFilter::update(const VectorXd& z, const MatrixXd& H,
                          const MatrixXd& R) {
  update_y(z - (H * x_), H, R);
}

void KalmanFilter::update_ekf(const VectorXd& z, const MatrixXd& H,
                              const MatrixXd& R) {
  double px = x_[0];
  double py = x_[1];
  double vx = x_[2];
  double vy = x_[3];

  // h(x): convert state vector to polar coordinates
  double rho = std::sqrt(px*px + py*py);
  if (is_nearly_zero(rho))
    return;
  double theta = std::atan2(py, px);
  double rho_dot = (px*vx + py*vy) / rho;

  VectorXd hx(3);
  hx << rho, theta, rho_dot;

  VectorXd y = z - hx;

  // Normalize theta of y to [-PI, PI]
  y[1] = std::atan2(std::sin(y[1]), std::cos(y[1]));

  update_y(y, H, R);
}

void KalmanFilter::update_y(const Eigen::VectorXd& y, const Eigen::MatrixXd& H,
                            const Eigen::MatrixXd& R) {
  MatrixXd Ht = H.transpose();
  MatrixXd PHt = P_ * Ht;
  MatrixXd S = H * PHt + R;
  MatrixXd K = PHt * S.inverse();
  x_ = x_ + (K * y);
  P_ = (I_ - K * H) * P_;
}
