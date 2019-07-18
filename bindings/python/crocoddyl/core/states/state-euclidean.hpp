///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_CORE_STATES_STATE_EUCLIDEAN_HPP_
#define PYTHON_CROCODDYL_CORE_STATES_STATE_EUCLIDEAN_HPP_

#include "crocoddyl/core/states/state-euclidean.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeStateEuclidean() {
  bp::class_<StateVector, bp::bases<StateAbstract>>("StateVector",
                                                    R"(Euclidean state vector.

        For this type of states, the difference and integrate operators are described by
        arithmetic subtraction and addition operations, respectively. Due to the Euclidean
        point and its velocity lie in the same space, all Jacobians are described throught
        the identity matrix.)",
                                                    bp::init<int>(bp::args(" self", " nx"),
                                                                  R"(Initialize the vector dimension.

:param nx: dimension of state)"))
      .def("zero", &StateVector::zero, bp::args(" self"),
           R"(Return a zero reference state.

:return zero reference state)")
      .def("rand", &StateVector::rand, bp::args(" self"),
           R"(Return a random reference state.

:return random reference state)")
      .def("diff", &StateVector::diff_wrap, bp::args(" self", " x0", " x1"),
           R"(Operator that differentiates the two state points.

It returns the value of x1 [-] x0 operation. Due to a state vector lies in
the Euclidean space, this operator is defined with arithmetic subtraction.
:param x0: current state (dim state.nx()).
:param x1: next state (dim state.nx()).
:return x1 - x0 value (dim state.nx()).)")
      .def("integrate", &StateVector::integrate_wrap, bp::args(" self", " x", " dx"),
           R"(Operator that integrates the current state.

It returns the value of x [+] dx operation. Due to a state vector lies in
the Euclidean space, this operator is defined with arithmetic addition.
Futhermore there is no timestep here (i.e. dx = v*dt), note this if you're
integrating a velocity v during an interval dt.
:param x: current state (dim state.nx()).
:param dx: displacement of the state (dim state.nx()).
:return x + dx value (dim state.nx()).)")
      .def("Jdiff", &StateVector::Jdiff_wrap,
           Jdiffs(bp::args(" self", " x0", " x1", " firstsecond = 'both'"),
                  R"(Compute the partial derivatives of arithmetic substraction.

Both Jacobian matrices are represented throught an identity matrix, with the exception
that the first partial derivatives (w.r.t. x0) has negative signed. By default, this
function returns the derivatives of the first and second argument (i.e.
firstsecond='both'). However we ask for a specific partial derivative by setting
firstsecond='first' or firstsecond='second'.
:param x0: current state (dim state.nx()).
:param x1: next state (dim state.nx()).
:param firstsecond: desired partial derivative
:return the partial derivative(s) of the diff(x0, x1) function)"))
      .def("Jintegrate", &StateVector::Jintegrate_wrap,
           Jintegrates(bp::args(" self", " x", " dx", " firstsecond = 'both'"),
                       R"(Compute the partial derivatives of arithmetic addition.

Both Jacobian matrices are represented throught an identity matrix. By default, this
function returns the derivatives of the first and second argument (i.e.
firstsecond='both'). However we ask for a specific partial derivative by setting
firstsecond='first' or firstsecond='second'.
:param x: current state (dim state.nx()).
:param dx: displacement of the state (dim state.nx()).
:param firstsecond: desired partial derivative
:return the partial derivative(s) of the integrate(x, dx) function)"));
}

}  // namespace python
}  // namespace crocoddyl

#endif  // PYTHON_CROCODDYL_CORE_STATES_STATE_EUCLIDEAN_HPP_