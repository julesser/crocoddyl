///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATION_BASE_HPP_
#define CROCODDYL_CORE_ACTIVATION_BASE_HPP_

#include <stdexcept>
#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include "crocoddyl/core/utils/to-string.hpp"

namespace crocoddyl {

struct ActivationDataAbstract;  // forward declaration

class ActivationModelAbstract {
 public:
  explicit ActivationModelAbstract(const std::size_t& nr);
  virtual ~ActivationModelAbstract();

  virtual void calc(const boost::shared_ptr<ActivationDataAbstract>& data,
                    const Eigen::Ref<const Eigen::VectorXd>& r) = 0;
  virtual void calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data,
                        const Eigen::Ref<const Eigen::VectorXd>& r) = 0;
  virtual boost::shared_ptr<ActivationDataAbstract> createData();

  const std::size_t& get_nr() const;

 protected:
  std::size_t nr_;

#ifdef PYTHON_BINDINGS

 public:
  void calc_wrap(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::VectorXd& r) { calc(data, r); }

  void calcDiff_wrap(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::VectorXd& r) {
    calcDiff(data, r);
  }

#endif
};

struct ActivationDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Activation>
  explicit ActivationDataAbstract(Activation* const activation)
      : a_value(0.),
        Ar(Eigen::VectorXd::Zero(activation->get_nr())),
        Arr(Eigen::MatrixXd::Zero(activation->get_nr(), activation->get_nr())) {}
  virtual ~ActivationDataAbstract() {}

  double a_value;
  Eigen::VectorXd Ar;
  Eigen::MatrixXd Arr;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATION_BASE_HPP_
