///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIONS_UNICYCLE_HPP_
#define CROCODDYL_CORE_ACTIONS_UNICYCLE_HPP_

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include <stdexcept>

namespace crocoddyl {

class ActionModelUnicycle : public ActionModelAbstract {
 public:
  ActionModelUnicycle();
  ~ActionModelUnicycle();

  void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u);
  boost::shared_ptr<ActionDataAbstract> createData();

  const Eigen::Vector2d& get_cost_weights() const;
  void set_cost_weights(const Eigen::Vector2d& weights);

 private:
  Eigen::Vector2d cost_weights_;
  double dt_;
};

struct ActionDataUnicycle : public ActionDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  explicit ActionDataUnicycle(Model* const model) : ActionDataAbstract(model) {}
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIONS_UNICYCLE_HPP_
