
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_FRAMES_HPP_
#define CROCODDYL_MULTIBODY_FRAMES_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/friction-cone.hpp"
#include "crocoddyl/core/mathbase.hpp"

#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/spatial/motion.hpp>
#include <pinocchio/spatial/force.hpp>

namespace crocoddyl {

typedef std::size_t FrameIndex;

template <typename _Scalar>
struct FrameTranslationTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef typename MathBaseTpl<Scalar>::Vector3s Vector3s;

  explicit FrameTranslationTpl() : frame(0), oxf(Vector3s::Zero()) {}
  FrameTranslationTpl(const FrameTranslationTpl<Scalar>& value) : frame(value.frame), oxf(value.oxf) {}
  FrameTranslationTpl(const FrameIndex& frame, const Vector3s& oxf) : frame(frame), oxf(oxf) {}
  friend std::ostream& operator<<(std::ostream& os, const FrameTranslationTpl<Scalar>& X) {
    os << "      frame: " << X.frame << std::endl << "translation: " << std::endl << X.oxf.transpose() << std::endl;
    return os;
  }

  FrameIndex frame;
  Vector3s oxf;
};

template <typename _Scalar>
struct FrameRotationTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef typename MathBaseTpl<Scalar>::Matrix3s Matrix3s;

  explicit FrameRotationTpl() : frame(0), oRf(Matrix3s::Identity()) {}
  FrameRotationTpl(const FrameRotationTpl<Scalar>& value) : frame(value.frame), oRf(value.oRf) {}
  FrameRotationTpl(const FrameIndex& frame, const Matrix3s& oRf) : frame(frame), oRf(oRf) {}
  friend std::ostream& operator<<(std::ostream& os, const FrameRotationTpl<Scalar>& X) {
    os << "   frame: " << X.frame << std::endl << "rotation: " << std::endl << X.oRf << std::endl;
    return os;
  }

  FrameIndex frame;
  Matrix3s oRf;
};

template <typename _Scalar>
struct FramePlacementTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef pinocchio::SE3Tpl<Scalar> SE3;

  explicit FramePlacementTpl() : frame(0), oMf(SE3::Identity()) {}
  FramePlacementTpl(const FramePlacementTpl<Scalar>& value) : frame(value.frame), oMf(value.oMf) {}
  FramePlacementTpl(const FrameIndex& frame, const SE3& oMf) : frame(frame), oMf(oMf) {}
  friend std::ostream& operator<<(std::ostream& os, const FramePlacementTpl<Scalar>& X) {
    os << "    frame: " << X.frame << std::endl << "placement: " << std::endl << X.oMf << std::endl;
    return os;
  }

  FrameIndex frame;
  pinocchio::SE3Tpl<Scalar> oMf;
};

template <typename _Scalar>
struct FrameMotionTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef pinocchio::MotionTpl<Scalar> Motion;

  explicit FrameMotionTpl() : frame(0), oMf(Motion::Zero()) {}
  FrameMotionTpl(const FrameMotionTpl<Scalar>& value) : frame(value.frame), oMf(value.oMf) {}
  FrameMotionTpl(const FrameIndex& frame, const Motion& oMf) : frame(frame), oMf(oMf) {}
  friend std::ostream& operator<<(std::ostream& os, const FrameMotionTpl<Scalar>& X) {
    os << " frame: " << X.frame << std::endl << "motion: " << std::endl << X.oMf << std::endl;
    return os;
  }

  FrameIndex frame;
  pinocchio::MotionTpl<Scalar> oMf;
};

template <typename _Scalar>
struct FrameForceTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef pinocchio::ForceTpl<Scalar> Force;

  explicit FrameForceTpl() : frame(0), oFf(Force::Zero()) {}
  FrameForceTpl(const FrameForceTpl<Scalar>& value) : frame(value.frame), oFf(value.oFf) {}
  FrameForceTpl(const FrameIndex& frame, const Force& oFf) : frame(frame), oFf(oFf) {}
  friend std::ostream& operator<<(std::ostream& os, const FrameForceTpl<Scalar>& X) {
    os << "frame: " << X.frame << std::endl << "force: " << std::endl << X.oFf << std::endl;
    return os;
  }

  FrameIndex frame;
  pinocchio::ForceTpl<Scalar> oFf;
};

template <typename _Scalar>
struct FrameFrictionConeTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef FrictionConeTpl<Scalar> FrictionCone;

  explicit FrameFrictionConeTpl() : frame(0), oRf(FrictionCone()) {}
  FrameFrictionConeTpl(const FrameFrictionConeTpl<Scalar>& value) : frame(value.frame), oRf(value.oRf) {}
  FrameFrictionConeTpl(const FrameIndex& frame, const FrictionCone& oRf) : frame(frame), oRf(oRf) {}
  friend std::ostream& operator<<(std::ostream& os, const FrameFrictionConeTpl& X) {
    os << "frame: " << X.frame << std::endl << " cone: " << std::endl << X.oRf << std::endl;
    return os;
  }

  FrameIndex frame;
  FrictionCone oRf;
};

template <typename _Scalar>
class FrameCoPSupportTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef typename MathBaseTpl<Scalar>::Vector2s Vector2s;
  typedef typename MathBaseTpl<Scalar>::Vector3s Vector3s;
  typedef Eigen::Matrix<Scalar, 4, 6> Matrix46;

 public:
  explicit FrameCoPSupportTpl() : frame_(0), support_region_(Vector2s::Zero()) { update_A(); }
  FrameCoPSupportTpl(const FrameCoPSupportTpl<Scalar>& value)
      : frame_(value.get_frame()), support_region_(value.get_support_region()), A_(value.get_A()) {}
  FrameCoPSupportTpl(const FrameIndex& frame, const Vector2s& support_region)
      : frame_(frame), support_region_(support_region) {
    update_A();
  }
  friend std::ostream& operator<<(std::ostream& os, const FrameCoPSupportTpl<Scalar>& X) {
    os << "          frame: " << X.get_frame() << std::endl
       << "foot dimensions: " << std::endl
       << X.get_support_region() << std::endl;
    return os;
  }

  // Define the inequality matrix A to implement A * f >= 0. Compare eq.(18-19) in
  // https://hal.archives-ouvertes.fr/hal-02108449/document
  void update_A() {
    A_ << Scalar(0), Scalar(0), support_region_[0] / Scalar(2), Scalar(0), Scalar(-1), Scalar(0), Scalar(0), Scalar(0),
        support_region_[0] / Scalar(2), Scalar(0), Scalar(1), Scalar(0), Scalar(0), Scalar(0),
        support_region_[1] / Scalar(2), Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
        support_region_[1] / Scalar(2), Scalar(-1), Scalar(0), Scalar(0);
  }

  void set_frame(FrameIndex frame) { frame_ = frame; }
  void set_support_region(const Vector2s& support_region) {
    support_region_ = support_region;
    update_A();
  }

  const FrameIndex& get_frame() const { return frame_; }
  const Vector2s& get_support_region() const { return support_region_; }
  const Matrix46& get_A() const { return A_; }

 private:
  FrameIndex frame_;         //!< contact frame ID
  Vector2s support_region_;  //!< cop support region = (length, width)
  Matrix46 A_;               //!< inequality matrix
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_FRAMES_HPP_
