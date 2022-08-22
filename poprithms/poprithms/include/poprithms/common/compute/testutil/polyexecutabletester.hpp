// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_TESTUTIL_POLYEXECUTABLETESTER_HPP
#define POPRITHMS_COMMON_COMPUTE_TESTUTIL_POLYEXECUTABLETESTER_HPP

#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include <poprithms/common/compute/iexecutable.hpp>
#include <poprithms/common/compute/simexecutable.hpp>

namespace poprithms {
namespace common {
namespace compute {
namespace testutil {

using poprithms::common::compute::IExecutable;

/**
 * An abstract class for running numerical tests that require an executable.
 * This class is useful for running the same test with different
 * implementations of the IExectuable class.
 **/
class PolyExecutableTester {

public:
  PolyExecutableTester()          = default;
  virtual ~PolyExecutableTester() = default;

  void setCompiledSlickGraph(Graph &g) {
    g.verifyValid();
    uptrCompiledGraph = getCompiledSlickGraph(g);
  }

  IExecutable &cm() { return *uptrCompiledGraph; }

  static void localAssert(bool b, const std::string &x) {
    if (!b) {
      throw poprithms::test::error("Local assert failed: " + x);
    }
  }

private:
  /**
   * \return The IExecutable to run a test on.
   * */
  virtual std::unique_ptr<IExecutable> getCompiledSlickGraph(Graph &m) = 0;

  std::unique_ptr<IExecutable> uptrCompiledGraph;

private:
  virtual void noWeakVTables();
};

template <typename BaseTester> class SimTester : public BaseTester {

public:
  SimTester()                   = default;
  virtual ~SimTester() override = default;

private:
  std::unique_ptr<IExecutable> getCompiledSlickGraph(Graph &m) final {
    return std::unique_ptr<poprithms::common::compute::SimExecutable>(
        new SimExecutable(m));
  }
};

} // namespace testutil
} // namespace compute
} // namespace common
} // namespace poprithms

#endif
