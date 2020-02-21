///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_STATE_HPP_
#define CROCODDYL_MULTIBODY_COSTS_STATE_HPP_

#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/multibody/cost-base.hpp"

namespace crocoddyl {

class CostModelState : public CostModelAbstract {
 public:
  CostModelState(boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
                 const Eigen::VectorXd& xref, const std::size_t& nu);
  CostModelState(boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
                 const Eigen::VectorXd& xref);
  CostModelState(boost::shared_ptr<StateMultibody> state, const Eigen::VectorXd& xref, const std::size_t& nu);
  CostModelState(boost::shared_ptr<StateMultibody> state, const Eigen::VectorXd& xref);
  CostModelState(boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
                 const std::size_t& nu);
  CostModelState(boost::shared_ptr<StateMultibody> state, const std::size_t& nu);
  CostModelState(boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation);
  explicit CostModelState(boost::shared_ptr<StateMultibody> state);

  ~CostModelState();

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u);
  boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

  const Eigen::VectorXd& get_xref() const;
  void set_xref(const Eigen::VectorXd& xref_in);

 private:
  Eigen::VectorXd xref_;
};

struct CostDataState : public CostDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  CostDataState(Model* const model, DataCollectorAbstract* const data)
      : CostDataAbstract(model, data), Arr_Rx(model->get_activation()->get_nr(), model->get_state()->get_ndx()) {
    Arr_Rx.fill(0);
  }

  Eigen::MatrixXd Arr_Rx;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_STATE_HPP_
