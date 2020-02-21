///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_HPP_

#include "crocoddyl/core/activations/weighted-quadratic.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeActivationWeightedQuad() {
  bp::class_<ActivationModelWeightedQuad, bp::bases<ActivationModelAbstract> >(
      "ActivationModelWeightedQuad",
      "Weighted quadratic activation model.\n\n"
      "A weighted quadratic action describes a quadratic function that depends on the residual,\n"
      "i.e. 0.5 *||r||_w^2.",
      bp::init<Eigen::VectorXd>(bp::args("self", "weights"),
                                "Initialize the activation model.\n\n"
                                ":param weights: weights vector, note that nr=weights.size()"))
      .def("calc", &ActivationModelWeightedQuad::calc_wrap, bp::args("self", "data", "r"),
           "Compute the 0.5 * ||r||_w^2.\n\n"
           ":param data: activation data\n"
           ":param r: residual vector")
      .def<void (ActivationModelWeightedQuad::*)(const boost::shared_ptr<ActivationDataAbstract>&,
                                                 const Eigen::VectorXd&)>(
          "calcDiff", &ActivationModelWeightedQuad::calcDiff_wrap, bp::args("self", "data", "r"),
          "Compute the derivatives of a quadratic function.\n\n"
          ":param data: activation data\n"
          "Note that the Hessian is constant, so we don't write again this value.\n"
          ":param r: residual vector \n")
      .def("createData", &ActivationModelWeightedQuad::createData, bp::args("self"),
           "Create the weighted quadratic action data.")
      .add_property(
          "weights",
          bp::make_function(&ActivationModelWeightedQuad::get_weights, bp::return_value_policy<bp::return_by_value>()),
          &ActivationModelWeightedQuad::set_weights, "weights of the quadratic term");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_HPP_
