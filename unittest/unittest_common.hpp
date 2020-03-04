///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

/**
 * To be included last in the test_XXX.cpp,
 * otherwise it interferes with pinocchio boost::variant.
 */

#ifndef CROCODDYL_UNITTEST_COMMON_HPP_
#define CROCODDYL_UNITTEST_COMMON_HPP_

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <iterator>
#include <Eigen/Dense>
#include <boost/bind.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/execution_monitor.hpp>  // for execution_exception
#include <boost/function.hpp>

#if __cplusplus >= 201103L
#include <random>
std::mt19937 rng;
#else
#include <boost/random.hpp>
#include <boost/nondet_random.hpp>
boost::random::mt19937 rng;
#endif

#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <string>
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl_unit_test {

class CaptureIOStream {
 public:
  CaptureIOStream() : m_oldStdOut(0), m_oldStdErr(0), m_capturing(false), m_init(false) {
    m_pipe[READ] = 0;
    m_pipe[WRITE] = 0;
    if (pipe(m_pipe) == -1) {
      throw_pretty("Cannot create pipe.");
    }
    m_oldStdOut = dup(fileno(stdout));
    m_oldStdErr = dup(fileno(stderr));
    if (m_oldStdOut == -1 || m_oldStdErr == -1) {
      throw_pretty("Cannot redirect stdout or stderr.");
    }

    m_init = true;
  }

  ~CaptureIOStream() {
    if (m_capturing) {
      endCapture();
    }
    if (m_oldStdOut > 0) close(m_oldStdOut);
    if (m_oldStdErr > 0) close(m_oldStdErr);
    if (m_pipe[READ] > 0) close(m_pipe[READ]);
    if (m_pipe[WRITE] > 0) close(m_pipe[WRITE]);
  }

  void beginCapture() {
    if (!m_init) return;
    if (m_capturing) endCapture();
    fflush(stdout);
    fflush(stderr);
    dup2(m_pipe[WRITE], fileno(stdout));
    dup2(m_pipe[WRITE], fileno(stderr));
    m_capturing = true;
  }

  bool endCapture() {
    usleep(2000);
    if (!m_init || !m_capturing) {
      return false;
    }
    fflush(stdout);
    fflush(stderr);
    dup2(m_oldStdOut, fileno(stdout));
    dup2(m_oldStdErr, fileno(stderr));
    m_captured.clear();

    // read the pipe
    ssize_t nb_read = 0;
    // Set timeout to 0.2 seconds
    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 200000;
    // Initialize file descriptor sets
    fd_set read_fds, write_fds, except_fds;
    FD_ZERO(&read_fds);
    FD_ZERO(&write_fds);
    FD_ZERO(&except_fds);
    FD_SET(m_pipe[READ], &read_fds);
    //
    bool timed_out = false;

    while (!timed_out) {
      if (select(m_pipe[READ] + 1, &read_fds, &write_fds, &except_fds, &timeout) == 1) {
        // do the reading
        char buff[1];
        nb_read = read(m_pipe[READ], buff, sizeof(buff));
        if (nb_read > 0) {
          m_captured << *buff;
        }
        timed_out = false;
      } else {
        // timeout or error
        timed_out = true;
      }
    }
    return true;
  }

  std::string str() const { return m_captured.str(); }

 private:
  enum PIPES { READ, WRITE };
  int m_pipe[2];
  int m_oldStdOut;
  int m_oldStdErr;
  bool m_capturing;
  bool m_init;
  std::ostringstream m_captured;
};

std::string GetErrorMessages(boost::function<int(void)> function_with_errors) {
  CaptureIOStream capture_ios;
  capture_ios.beginCapture();
  boost::execution_monitor monitor;
  try {
    monitor.execute(function_with_errors);
  } catch (...) {
  }
  capture_ios.endCapture();
  return capture_ios.str();
}

template <typename IntType>
IntType random_int_in_range(IntType first = 0, IntType last = 10) {
#if __cplusplus >= 201103L
  return std::uniform_int_distribution<IntType>(first, last)(rng);
#else
  return boost::random::uniform_int_distribution<IntType>(first, last)(rng);
#endif
}

template <typename RealType>
RealType random_real_in_range(RealType first = 0, RealType last = 1) {
#if __cplusplus >= 201103L
  return std::uniform_real_distribution<RealType>(first, last)(rng);
#else
  return boost::random::uniform_real_distribution<RealType>(first, last)(rng);
#endif
}

}  // namespace crocoddyl_unit_test

#endif  // CROCODDYL_UNITTEST_COMMON_HPP_
