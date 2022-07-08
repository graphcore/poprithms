// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <iostream>
#include <sstream>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/ops/unaryelementwise.hpp>
#include <poprithms/common/compute/simexecutable.hpp>
#include <poprithms/common/compute/slickgraph.hpp>
#include <poprithms/compute/host/tensor.hpp>

namespace {
using namespace poprithms::common::compute;

// Example transformation which uses the identity
//
//       sin**2 + cos**2 = 1,          (1)
//
// to replace,
//       abs(sin(x)),                  (2)
//
// in the graph with,
//       sqrt((1 - cos(x)*cos(x)))     (3).
//
bool expressSinAsCos(SlickGraph &graph) {

  bool applied{false};

  // for all remaining sin ops in the graph:
  for (auto sinId : graph.opIds()) {
    if (graph.isLive(sinId) && graph.dynamicCast<Sin>(sinId) &&
        !graph.hasDerivedRefs({sinId, 0})) {

      // if the output of the sin op has 1 consumer, and it's an abs (pattern
      // matching for (2)):
      auto consumers = graph.consumptionIds({sinId, 0});
      if (consumers.size() == 1 &&
          graph.dynamicCast<Abs>(consumers[0].opId())) {

        // construct the alternative path (3).
        const auto c =
            graph.tensor(graph.inTensorId(sinId, InIndex(0))).cos();
        const auto x = (c.constant(1.) - c * c).sqrt();

        // NB: If there are control dependencies to transfer, they should be
        // propagated here.We don't transfer control deps by default.
        const auto absId = consumers[0].opId();
        graph.removeOp(absId, {x.id()}, "expressSinAsCos, removing Abs");
        graph.removeOp(sinId, {{}}, "expressSinAsCos, removing Sin");
        applied = true;
      }
    }
  }
  return applied;
}

void numericalTest() {

  SlickGraph graph;
  auto sg = graph.createSubGraph("sg0");
  const Shape shape{7, 5};

  const auto var0 = sg.variable(DType::Float32, shape, graph.host());
  const auto foo  = var0.sin().abs();
  const auto out  = foo.exp() + foo.abs();

  const auto x0 = HostTensor::uniformFloat32(-4, +4, shape, 1011);

  graph.setRunnable({sg.id()});

  auto getValue = [&graph, sg, out, var0, x0]() {
    SimExecutable cm(graph);
    cm.setHostValue(var0.id(), x0);
    cm.run(sg.id());
    return cm.getHostValue(out.id());
  };

  auto preTransformValue = getValue();

  auto applied = expressSinAsCos(graph);
  if (!applied) {
    throw poprithms::test::error(
        "Failed to apply the test transformation sinAsCos");
  }
  auto postTransformValue = getValue();

  const auto maxError = (postTransformValue - preTransformValue)
                            .abs()
                            .reduceMax()
                            .getFloat64(0);
  if (maxError > 1e-4) {
    std::cout << maxError << std::endl;
    throw poprithms::test::error(
        "Numerical error too high in transform test");
  }

  graph.verifyValid();
}

void extendedTest() {

  SlickGraph graph;
  auto sg0     = graph.createSubGraph("sg0");
  const auto x = sg0.hostInt32Variable({}).sin().abs();

  auto sg1 = graph.createSubGraph("sg1");
  auto r   = x.refTo_(sg1);

  expressSinAsCos(graph);
  auto asSqrt = graph.dynamicCast<Sqrt>(r.rootRef().opId());
  if (!asSqrt) {
    std::ostringstream oss;
    oss << "Expected the root reference to change to a the final op in the "
           "transform chain";
    throw poprithms::test::error(oss.str());
  }

  graph.verifyValid();
}

// A further example of a transform, which simply removes all sin ops.
void standAloneRemoveSinExample() {
  auto removeSin = [](Graph &machine) {
    for (auto opId : machine.opIds()) {
      if (auto asSin = machine.dynamicCast<Sin>(opId)) {
        machine.removeOp(opId, {asSin->inTensorId(0)}, "removeSin");
      }
    }
  };

  SlickGraph m;

  const auto devId = m.host();
  auto sg          = m.createSubGraph("sg0");
  const auto out   = sg.variable(DType::Float32, {3, 2}, devId)
                       .sin()
                       .sin()
                       .exp()
                       .abs()
                       .sin();
  for (uint64_t i = 0; i < 3; ++i) {
    out.flatten_().abs();
  }

  removeSin(m);
  m.verifyValid();
  if (m.opIds<Sin>().size() != 0) {
    throw poprithms::test::error("Expected all sin ops to be removed");
  }
}

} // namespace

int main() {
  numericalTest();
  extendedTest();
  standAloneRemoveSinExample();
  return 0;
}
