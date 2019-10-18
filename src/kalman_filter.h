#pragma once

#include "Eigen/Dense"

class KalmanFilter {
public:
  KalmanFilter() {}
  virtual ~KalmanFilter() {}

  // Initializes Kalman filter
  // @param x Initial state
  // @param P Initial state covariance
  void init(const Eigen::VectorXd& x, const Eigen::MatrixXd& P);

  // Predicts the state and the state covariance using the process model
  // @param F Transition matrix
  // @param Q Process covariance matrix
  void predict(const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q);

  // Updates the state by using standard Kalman Filter equations
  // @param z The measurement at k+1
  // @param H Measurement matrix
  // @param R Measurement covariance matrix
  void update(const Eigen::VectorXd& z, const Eigen::MatrixXd& H,
              const Eigen::MatrixXd& R);

  // Updates the state by using Extended Kalman Filter equations for polar
  // coordinate measurements
  // @param z The measurement at k+1
  // @param H Measurement matrix
  // @param R Measurement covariance matrix
  void update_ekf(const Eigen::VectorXd& z, const Eigen::MatrixXd& H,
                  const Eigen::MatrixXd& R);

  // State (read-only)
  const Eigen::VectorXd& x() { return x_; }
  const Eigen::MatrixXd& P() { return P_; }

private:
  void update_y(const Eigen::VectorXd& y, const Eigen::MatrixXd& H,
                const Eigen::MatrixXd& R);

  // State vector
  Eigen::VectorXd x_;

  // State covariance matrix
  Eigen::MatrixXd P_;
};
