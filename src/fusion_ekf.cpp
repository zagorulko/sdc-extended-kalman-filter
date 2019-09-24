#include "fusion_ekf.h"

#include <cassert>
#include <iostream>

#include "Eigen/Dense"

#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

FusionEKF::FusionEKF()
    : ekf_(),
      is_initialized_(false),
      previous_timestamp_(0),
      R_laser_(MatrixXd(2, 2)),
      R_radar_(MatrixXd(3, 3)),
      H_laser_(MatrixXd(2, 4)),
      Hj_(MatrixXd(3, 4)),
      F_(MatrixXd(4, 4)),
      Q_(MatrixXd(4, 4)) {
  // Measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;
  // Measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;
  // Measurement matrix - laser
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
}

FusionEKF::~FusionEKF() {}

void FusionEKF::process_measurement(const MeasurementPackage& measurement_pack) {
  if (!is_initialized_) {
    cout << "Initializing EKF" << endl;
    init(measurement_pack);
    is_initialized_ = true;
    return;
  }
  predict(measurement_pack);
  update(measurement_pack);
  cout << "x =" << endl << ekf_.x() << endl << endl;
  cout << "P =" << endl << ekf_.P() << endl << endl;
  cout << endl;
}

void FusionEKF::init(const MeasurementPackage& measurement_pack) {
  const auto& z = measurement_pack.raw_measurements_;
  double px, py;
  switch (measurement_pack.sensor_type_) {
  case MeasurementPackage::RADAR:
    // Convert from polar to cartesian coordinates
    px = z[0] * std::cos(z[1]);
    py = z[0] * std::sin(z[1]);
    break;
  case MeasurementPackage::LASER:
    px = z[0];
    py = z[1];
    break;
  default:
    assert(0);
    break;
  }

  // Initial state vector
  VectorXd x(4);
  x << px, py, 0, 0;

  // Initial state covariance matrix
  MatrixXd P(4, 4);
  P << 1, 0, 0, 0,
       0, 1, 0, 0,
       0, 0, 1000, 0,
       0, 0, 0, 1000;

  ekf_.init(x, P);

  previous_timestamp_ = measurement_pack.timestamp_;

  // Initial state transition matrix
  F_ << 1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1;
}

void FusionEKF::predict(const MeasurementPackage& measurement_pack) {
  double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  // Update the state transition matrix
  F_(0, 2) = dt;
  F_(1, 3) = dt;

  // Acceleration noise components
  double noise_ax = 9;
  double noise_ay = 9;

  // Precomputed terms
  double dt_2 = dt*dt;
  double dt_3 = dt_2*dt;
  double dt_4 = dt_3*dt;

  // Update the process noise covariance matrix
  Q_ << dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
        0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
        dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
        0, dt_3/2*noise_ay, 0, dt_2*noise_ay;

  ekf_.predict(F_, Q_);
}

void FusionEKF::update(const MeasurementPackage& measurement_pack) {
  const auto& z = measurement_pack.raw_measurements_;
  switch (measurement_pack.sensor_type_) {
  case MeasurementPackage::RADAR:
    Hj_ = calculate_jacobian(ekf_.x());
    ekf_.update_ekf(z, Hj_, R_radar_);
    break;
  case MeasurementPackage::LASER:
    ekf_.update(z, H_laser_, R_laser_);
    break;
  default:
    assert(0);
    break;
  }
}
