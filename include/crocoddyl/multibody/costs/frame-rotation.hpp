///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_FRAME_ROTATION_HPP_
#define CROCODDYL_MULTIBODY_COSTS_FRAME_ROTATION_HPP_

#include "crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

class CostModelFrameRotation : public CostModelAbstract {
 public:
  CostModelFrameRotation(boost::shared_ptr<StateMultibody> state,
                         boost::shared_ptr<ActivationModelAbstract> activation, const FrameRotation& Fref,
                         const std::size_t& nu);
  CostModelFrameRotation(boost::shared_ptr<StateMultibody> state,
                         boost::shared_ptr<ActivationModelAbstract> activation, const FrameRotation& Fref);
  CostModelFrameRotation(boost::shared_ptr<StateMultibody> state, const FrameRotation& Fref, const std::size_t& nu);
  CostModelFrameRotation(boost::shared_ptr<StateMultibody> state, const FrameRotation& Fref);
  ~CostModelFrameRotation();

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u);
  boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

  const FrameRotation& get_Rref() const;
  void set_Rref(const FrameRotation& Rref_in);

 private:
  FrameRotation Rref_;
  Eigen::Matrix3d oRf_inv_;
};

struct CostDataFrameRotation : public CostDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  CostDataFrameRotation(Model* const model, DataCollectorAbstract* const data)
      : CostDataAbstract(model, data),
        J(3, model->get_state()->get_nv()),
        rJf(3, 3),
        fJf(6, model->get_state()->get_nv()),
        Arr_J(3, model->get_state()->get_nv()) {
    r.fill(0);
    rRf.setIdentity();
    J.fill(0);
    rJf.fill(0);
    fJf.fill(0);
    Arr_J.fill(0);
    // Check that proper shared data has been passed
    DataCollectorMultibody* d = dynamic_cast<DataCollectorMultibody*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorMultibody");
    }

    // Avoids data casting at runtime
    pinocchio = d->pinocchio;
  }

  pinocchio::Data* pinocchio;
  Eigen::Vector3d r;
  Eigen::Matrix3d rRf;
  pinocchio::Data::Matrix3x J;
  Eigen::Matrix3d rJf;
  pinocchio::Data::Matrix6x fJf;
  pinocchio::Data::Matrix3x Arr_J;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_FRAME_ROTATION_HPP_
