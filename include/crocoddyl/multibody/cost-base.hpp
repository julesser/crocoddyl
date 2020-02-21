///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COST_BASE_HPP_
#define CROCODDYL_MULTIBODY_COST_BASE_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/core/data-collector-base.hpp"
#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/utils/to-string.hpp"

namespace crocoddyl {

struct CostDataAbstract;  // forward declaration

class CostModelAbstract {
 public:
  CostModelAbstract(boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
                    const std::size_t& nu, const bool& with_residuals = true);
  CostModelAbstract(boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
                    const bool& with_residuals = true);
  CostModelAbstract(boost::shared_ptr<StateMultibody> state, const std::size_t& nr, const std::size_t& nu,
                    const bool& with_residuals = true);
  CostModelAbstract(boost::shared_ptr<StateMultibody> state, const std::size_t& nr, const bool& with_residuals = true);
  ~CostModelAbstract();

  virtual void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                    const Eigen::Ref<const Eigen::VectorXd>& u) = 0;
  virtual void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                        const Eigen::Ref<const Eigen::VectorXd>& u) = 0;
  virtual boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x);

  const boost::shared_ptr<StateMultibody>& get_state() const;
  const boost::shared_ptr<ActivationModelAbstract>& get_activation() const;
  const std::size_t& get_nu() const;

 protected:
  boost::shared_ptr<StateMultibody> state_;
  boost::shared_ptr<ActivationModelAbstract> activation_;
  std::size_t nu_;
  bool with_residuals_;
  Eigen::VectorXd unone_;

#ifdef PYTHON_BINDINGS

 public:
  void calc_wrap(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::VectorXd& x,
                 const Eigen::VectorXd& u = Eigen::VectorXd()) {
    if (u.size() == 0) {
      calc(data, x);
    } else {
      calc(data, x, u);
    }
  }

  void calcDiff_wrap(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::VectorXd& x,
                     const Eigen::VectorXd& u) {
    calcDiff(data, x, u);
  }
  void calcDiff_wrap(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::VectorXd& x) {
    calcDiff(data, x, unone_);
  }

#endif
};

struct CostDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  CostDataAbstract(Model* const model, DataCollectorAbstract* const data)
      : shared(data),
        activation(model->get_activation()->createData()),
        cost(0.),
        Lx(model->get_state()->get_ndx()),
        Lu(model->get_nu()),
        Lxx(model->get_state()->get_ndx(), model->get_state()->get_ndx()),
        Lxu(model->get_state()->get_ndx(), model->get_nu()),
        Luu(model->get_nu(), model->get_nu()),
        r(model->get_activation()->get_nr()),
        Rx(model->get_activation()->get_nr(), model->get_state()->get_ndx()),
        Ru(model->get_activation()->get_nr(), model->get_nu()) {
    Lx.fill(0);
    Lu.fill(0);
    Lxx.fill(0);
    Lxu.fill(0);
    Luu.fill(0);
    r.fill(0);
    Rx.fill(0);
    Ru.fill(0);
  }
  virtual ~CostDataAbstract() {}

  DataCollectorAbstract* shared;
  boost::shared_ptr<ActivationDataAbstract> activation;
  double cost;
  Eigen::VectorXd Lx;
  Eigen::VectorXd Lu;
  Eigen::MatrixXd Lxx;
  Eigen::MatrixXd Lxu;
  Eigen::MatrixXd Luu;
  Eigen::VectorXd r;
  Eigen::MatrixXd Rx;
  Eigen::MatrixXd Ru;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COST_BASE_HPP_
