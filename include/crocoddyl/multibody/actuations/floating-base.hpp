///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTUATIONS_FLOATING_BASE_HPP_
#define CROCODDYL_MULTIBODY_ACTUATIONS_FLOATING_BASE_HPP_

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

class ActuationModelFloatingBase : public ActuationModelAbstract {
 public:
  explicit ActuationModelFloatingBase(boost::shared_ptr<StateMultibody> state);
  ~ActuationModelFloatingBase();

  void calc(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u);
  boost::shared_ptr<ActuationDataAbstract> createData();

#ifndef NDEBUG
 private:
  Eigen::MatrixXd dtau_du_;
#endif
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_ACTUATIONS_FLOATING_BASE_HPP_
