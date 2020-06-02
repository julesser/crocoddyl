///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/parsers/urdf.hpp>
#include <example-robot-data/path.hpp>
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/actions/free-fwddyn.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/multibody/costs/cost-sum.hpp"
#include "crocoddyl/multibody/costs/frame-placement.hpp"
#include "crocoddyl/multibody/costs/state.hpp"
#include "crocoddyl/multibody/costs/control.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/utils/timer.hpp"

int main(int argc, char* argv[]) {
  bool CALLBACKS = false;
  unsigned int N = 100;  // number of nodes
  unsigned int T = 5e3;  // number of trials
  unsigned int MAXITER = 1;
  if (argc > 1) {
    T = atoi(argv[1]);
  }

  pinocchio::Model model;
  pinocchio::urdf::buildModel(EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_left_arm.urdf", model);
  boost::shared_ptr<crocoddyl::StateMultibody> state =
      boost::make_shared<crocoddyl::StateMultibody>(boost::make_shared<pinocchio::Model>(model));

  Eigen::VectorXd q0(state->get_nq());
  Eigen::VectorXd x0(state->get_nx());
  q0 << 0.173046, 1., -0.52366, 0., 0., 0.1, -0.005;
  x0 << q0, Eigen::VectorXd::Zero(state->get_nv());

  // Note that we need to include a cost model (i.e. set of cost functions) in
  // order to fully define the action model for our optimal control problem.
  // For this particular example, we formulate three running-cost functions:
  // goal-tracking cost, state and control regularization; and one terminal-cost:
  // goal cost. First, let's create the common cost functions.
  crocoddyl::FramePlacement Mref(model.getFrameId("gripper_left_joint"),
                                 pinocchio::SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(.0, .0, .4)));
  boost::shared_ptr<crocoddyl::CostModelAbstract> goalTrackingCost =
      boost::make_shared<crocoddyl::CostModelFramePlacement>(state, Mref);
  boost::shared_ptr<crocoddyl::CostModelAbstract> xRegCost = boost::make_shared<crocoddyl::CostModelState>(state);
  boost::shared_ptr<crocoddyl::CostModelAbstract> uRegCost = boost::make_shared<crocoddyl::CostModelControl>(state);

  // Create a cost model per the running and terminal action model.
  boost::shared_ptr<crocoddyl::CostModelSum> runningCostModel = boost::make_shared<crocoddyl::CostModelSum>(state);
  boost::shared_ptr<crocoddyl::CostModelSum> terminalCostModel = boost::make_shared<crocoddyl::CostModelSum>(state);

  // Then let's added the running and terminal cost functions
  runningCostModel->addCost("gripperPose", goalTrackingCost, 1);
  runningCostModel->addCost("xReg", xRegCost, 1e-4);
  runningCostModel->addCost("uReg", uRegCost, 1e-4);
  terminalCostModel->addCost("gripperPose", goalTrackingCost, 1);

  // We define an actuation model
  boost::shared_ptr<crocoddyl::ActuationModelFull> actuation =
      boost::make_shared<crocoddyl::ActuationModelFull>(state);

  // Next, we need to create an action model for running and terminal knots. The
  // forward dynamics (computed using ABA) are implemented
  // inside DifferentialActionModelFullyActuated.
  boost::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics> runningDAM =
      boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(state, actuation, runningCostModel);
  boost::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics> terminalDAM =
      boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(state, actuation, terminalCostModel);
  Eigen::VectorXd armature(state->get_nq());
  armature << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.;
  runningDAM->set_armature(armature);
  terminalDAM->set_armature(armature);
  boost::shared_ptr<crocoddyl::ActionModelAbstract> runningModel =
      boost::make_shared<crocoddyl::IntegratedActionModelEuler>(runningDAM, 1e-3);
  boost::shared_ptr<crocoddyl::ActionModelAbstract> terminalModel =
      boost::make_shared<crocoddyl::IntegratedActionModelEuler>(terminalDAM, 1e-3);

  // For this optimal control problem, we define 100 knots (or running action
  // models) plus a terminal knot
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > runningModels(N, runningModel);
  boost::shared_ptr<crocoddyl::ShootingProblem> problem =
      boost::make_shared<crocoddyl::ShootingProblem>(x0, runningModels, terminalModel);
  std::vector<Eigen::VectorXd> xs(N + 1, x0);
  std::vector<Eigen::VectorXd> us(N, Eigen::VectorXd::Zero(runningModel->get_nu()));
  for (unsigned int i = 0; i < N; ++i) {
    const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = problem->get_runningModels()[i];
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = problem->get_runningDatas()[i];
    model->quasiStatic(data, us[i], x0);
  }

  // Formulating the optimal control problem
  crocoddyl::SolverDDP ddp(problem);
  if (CALLBACKS) {
    std::vector<boost::shared_ptr<crocoddyl::CallbackAbstract> > cbs;
    cbs.push_back(boost::make_shared<crocoddyl::CallbackVerbose>());
    ddp.setCallbacks(cbs);
  }

  // Solving the optimal control problem
  Eigen::ArrayXd duration(T);
  for (unsigned int i = 0; i < T; ++i) {
    crocoddyl::Timer timer;
    ddp.solve(xs, us, MAXITER, false, 0.1);
    duration[i] = timer.get_duration();
  }

  double avrg_duration = duration.sum() / T;
  double min_duration = duration.minCoeff();
  double max_duration = duration.maxCoeff();
  std::cout << "  DDP.solve [ms]: " << avrg_duration << " (" << min_duration << "-" << max_duration << ")"
            << std::endl;

  // Running calc
  for (unsigned int i = 0; i < T; ++i) {
    crocoddyl::Timer timer;
    problem->calc(xs, us);
    duration[i] = timer.get_duration();
  }

  avrg_duration = duration.sum() / T;
  min_duration = duration.minCoeff();
  max_duration = duration.maxCoeff();
  std::cout << "  ShootingProblem.calc [ms]: " << avrg_duration << " (" << min_duration << "-" << max_duration << ")"
            << std::endl;

  // Running calcDiff
  for (unsigned int i = 0; i < T; ++i) {
    crocoddyl::Timer timer;
    problem->calcDiff(xs, us);
    duration[i] = timer.get_duration();
  }

  avrg_duration = duration.sum() / T;
  min_duration = duration.minCoeff();
  max_duration = duration.maxCoeff();
  std::cout << "  ShootingProblem.calcDiff [ms]: " << avrg_duration << " (" << min_duration << "-" << max_duration
            << ")" << std::endl;
}
