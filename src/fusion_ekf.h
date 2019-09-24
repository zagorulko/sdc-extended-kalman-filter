#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "Eigen/Dense"

#include "kalman_filter.h"
#include "measurement_package.h"
#include "tools.h"

class FusionEKF {
public:
  FusionEKF();
  virtual ~FusionEKF();

  // Run the whole flow of the Kalman Filter from here
  void process_measurement(const MeasurementPackage& measurement_pack);

  // Kalman Filter update and prediction math lives in here
  KalmanFilter ekf_;

private:
  void init(const MeasurementPackage& measurement_pack);
  void predict(const MeasurementPackage& measurement_pack);
  void update(const MeasurementPackage& measurement_pack);

  bool is_initialized_;

  long long previous_timestamp_;

  Eigen::MatrixXd R_laser_;
  Eigen::MatrixXd R_radar_;
  Eigen::MatrixXd H_laser_;
  Eigen::MatrixXd Hj_;

  Eigen::MatrixXd F_;
  Eigen::MatrixXd Q_;
};
