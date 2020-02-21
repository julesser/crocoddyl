///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_HPP_

#include "crocoddyl/core/activations/quadratic.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeActivationQuad() {
  bp::class_<ActivationModelQuad, bp::bases<ActivationModelAbstract> >(
      "ActivationModelQuad",
      "Quadratic activation model.\n\n"
      "A quadratic action describes a quadratic function that depends on the residual, i.e.\n"
      "0.5 *||r||^2.",
      bp::init<int>(bp::args("self", "nr"),
                    "Initialize the activation model.\n\n"
                    ":param nr: dimension of the cost-residual vector"))
      .def("calc", &ActivationModelQuad::calc_wrap, bp::args("self", "data", "r"),
           "Compute the 0.5 * ||r||^2.\n\n"
           ":param data: activation data\n"
           ":param r: residual vector")
      .def<void (ActivationModelQuad::*)(const boost::shared_ptr<ActivationDataAbstract>&, const Eigen::VectorXd&)>(
          "calcDiff", &ActivationModelQuad::calcDiff_wrap, bp::args("self", "data", "r"),
          "Compute the derivatives of a quadratic function.\n\n"
          "Note that the Hessian is constant, so we don't write again this value.\n"
          ":param data: activation data\n"
          ":param r: residual vector \n")
      .def("createData", &ActivationModelQuad::createData, bp::args("self"),
           "Create the quadratic activation data.\n\n");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_HPP_
