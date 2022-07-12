// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef CEDAR_COMPUTE_MACHINE_FINITE_DIFFERENCE_HPP
#define CEDAR_COMPUTE_MACHINE_FINITE_DIFFERENCE_HPP

#include <sstream>

#include <poprithms/autodiff/testutil/finitedifference.hpp>
#include <poprithms/common/compute/iexecutable.hpp>
#include <poprithms/common/compute/tensor.hpp>

namespace poprithms {
namespace common {
namespace compute {
namespace testutil {

/**
 * A thin wrapper around poprithms::autodiff::testutil::Checker::check for a
 * single sub-graph in a compute::Graph which does a fwd-bwd pass (computes
 * loss and gradient).
 * */
template <typename T>
void finiteDifferenceTest(
    IExecutable &executable,
    const T &loss,
    const T &target,
    const T &targetGrad,
    const std::unordered_map<TensorId, HostTensor> &initVals,
    const uint32_t randomSeed     = 1011,
    const double perturbationSize = 1e-5) {

  if (targetGrad.info() != target.info()) {
    std::ostringstream oss;
    oss << "The provided gradient has info " << targetGrad.info()
        << ", the the tensor it is supposedly a gradient of has info "
        << target.info() << ". For this method, they must be identical.";
    throw poprithms::test::error(oss.str());
  }

  const auto fwdBwdSgId = target.subGraphId();

  // first run. Get the value of the gradient of the target which needs to be
  // tested with the finite difference method.
  executable.setHostValues(initVals);
  executable.run(fwdBwdSgId);
  const auto gradOut = executable.getHostValue(targetGrad).copy();

  // a function to compute the loss when target has value targetValue.
  auto getLoss =
      [&executable, target, loss, fwdBwdSgId](const HostTensor &targetValue) {
        executable.setHostValue(target, targetValue.copy());
        executable.run(fwdBwdSgId);
        return executable.getHostValue(loss).copy();
      };

  poprithms::autodiff::testutil::Checker::check(getLoss,
                                                initVals.at(target).copy(),
                                                gradOut,
                                                perturbationSize,
                                                randomSeed);
}

} // namespace testutil
} // namespace compute
} // namespace common
} // namespace poprithms

#endif
