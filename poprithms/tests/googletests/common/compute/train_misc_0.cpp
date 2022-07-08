// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <gmock/gmock.h>
#include <sstream>

#include <poprithms/autodiff/testutil/finitedifference.hpp>
#include <poprithms/common/compute/autodiff/autodiffer.hpp>
#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/ops/init.hpp>
#include <poprithms/common/compute/ops/unaryelementwise.hpp>
#include <poprithms/common/compute/ops/viewchange.hpp>
#include <poprithms/common/compute/scheduler.hpp>
#include <poprithms/common/compute/simexecutable.hpp>
#include <poprithms/common/compute/slickgraph.hpp>

namespace {

using namespace poprithms::common::compute;
using Ad = Autodiffer<SlickGraph, AutomaticMutator>;

} // namespace

// We check that recomputation happens when only the inputs are checkpointed.
TEST(CommonComputeTrainMisc0, Recompute) {

  SlickGraph graph;

  /**
   * (1)    out = sqrt(sin(in) + 2).
   *
   * (2)    dOut = 1/(sqrt(sin(in) + 2)) * (1/2) * cos(in).
   * */
  auto sgFwd = graph.createSubGraph("fwd");
  auto d     = sgFwd.variable(DType::Float64, {2, 2}, graph.host()).name("d");
  auto c     = d.constant(2.0);
  auto out   = (d.sin() + c).sqrt().reduceSum(Shape({}));

  Ad ad(graph);
  auto sgBwdId = ad.backwardOutOfGraph(
      /* gradsProvidedFor = */ {out},
      /* checkpoints      = */ {d},
      /* requiresGrad     = */ {d});

  SubGraph sgBwd(sgBwdId, graph);

  auto &&gi = ad.gradInfo(sgBwd);

  // Expect the sin to be run for recomputation, too.
  EXPECT_EQ(graph.opIds<Sin>().size(), 2);

  graph.setRunnable({sgFwd, sgBwd});

  SimExecutable se(graph);

  // Compute the gradient of d0 using sbBwd:
  auto d0 = HostTensor::float64({2, 2}, {1, 2, 3, 4});
  se.setHostValue(gi.checkpointInGradGraph(d), d0);
  se.setHostValue(gi.gradInputInGradGraph(out), HostTensor::float64(1));
  se.run(sgBwd);
  auto g0 = se.getHostValue(gi.targetGradInGradGraph(d));

  // Perform finite-difference method to confirm the gradient is correct:
  auto fwd = [&](const HostTensor &ht) {
    se.setHostValue(d, ht);
    se.run(sgFwd);
    auto v = se.getHostValue(out).copy();
    return v;
  };
  double perturbationSize{0.001};
  uint64_t seed{1011};
  double eps0{1e-10};
  double threshold{1e-5};
  poprithms::autodiff::testutil::Checker::check(
      fwd, d0.copy(), g0, perturbationSize, seed, eps0, threshold);

  // We can also check the gradient against the derivation (2) above.
  auto expected = (d0.sin().add(2)).sqrt().pow(-1).mul(0.5).mul(d0.cos());
  g0.assertAllClose(expected, 1e-6, 1e-6, "compare to hand-derived gradient");
}
