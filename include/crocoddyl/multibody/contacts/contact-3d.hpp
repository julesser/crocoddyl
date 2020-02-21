///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_CONTACTS_CONTACT_3D_HPP_
#define CROCODDYL_MULTIBODY_CONTACTS_CONTACT_3D_HPP_

#include <pinocchio/spatial/motion.hpp>
#include <pinocchio/multibody/data.hpp>
#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/frames.hpp"

namespace crocoddyl {

class ContactModel3D : public ContactModelAbstract {
 public:
  ContactModel3D(boost::shared_ptr<StateMultibody> state, const FrameTranslation& xref, const std::size_t& nu,
                 const Eigen::Vector2d& gains = Eigen::Vector2d::Zero());
  ContactModel3D(boost::shared_ptr<StateMultibody> state, const FrameTranslation& xref,
                 const Eigen::Vector2d& gains = Eigen::Vector2d::Zero());
  ~ContactModel3D();

  void calc(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x);
  void calcDiff(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x);
  void updateForce(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::VectorXd& force);
  boost::shared_ptr<ContactDataAbstract> createData(pinocchio::Data* const data);

  const FrameTranslation& get_xref() const;
  const Eigen::Vector2d& get_gains() const;

 private:
  FrameTranslation xref_;
  Eigen::Vector2d gains_;
};

struct ContactData3D : public ContactDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  ContactData3D(Model* const model, pinocchio::Data* const data)
      : ContactDataAbstract(model, data),
        fJf(6, model->get_state()->get_nv()),
        v_partial_dq(6, model->get_state()->get_nv()),
        a_partial_dq(6, model->get_state()->get_nv()),
        a_partial_dv(6, model->get_state()->get_nv()),
        a_partial_da(6, model->get_state()->get_nv()),
        fXjdv_dq(6, model->get_state()->get_nv()),
        fXjda_dq(6, model->get_state()->get_nv()),
        fXjda_dv(6, model->get_state()->get_nv()) {
    frame = model->get_xref().frame;
    joint = model->get_state()->get_pinocchio().frames[frame].parent;
    jMf = model->get_state()->get_pinocchio().frames[frame].placement;
    fXj = jMf.inverse().toActionMatrix();
    fJf.fill(0);
    v_partial_dq.fill(0);
    a_partial_dq.fill(0);
    a_partial_dv.fill(0);
    a_partial_da.fill(0);
    fXjdv_dq.fill(0);
    fXjda_dq.fill(0);
    fXjda_dv.fill(0);
    vv.fill(0);
    vw.fill(0);
    vv_skew.fill(0);
    vw_skew.fill(0);
    oRf.fill(0);
  }

  pinocchio::Motion v;
  pinocchio::Motion a;
  pinocchio::Data::Matrix6x fJf;
  pinocchio::Data::Matrix6x v_partial_dq;
  pinocchio::Data::Matrix6x a_partial_dq;
  pinocchio::Data::Matrix6x a_partial_dv;
  pinocchio::Data::Matrix6x a_partial_da;
  pinocchio::Data::Matrix6x fXjdv_dq;
  pinocchio::Data::Matrix6x fXjda_dq;
  pinocchio::Data::Matrix6x fXjda_dv;
  Eigen::Vector3d vv;
  Eigen::Vector3d vw;
  Eigen::Matrix3d vv_skew;
  Eigen::Matrix3d vw_skew;
  Eigen::Matrix3d oRf;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_CONTACTS_CONTACT_3D_HPP_
