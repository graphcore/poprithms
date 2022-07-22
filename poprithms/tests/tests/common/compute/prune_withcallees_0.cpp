// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <numeric>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/ops/binaryelementwise.hpp>
#include <poprithms/common/compute/ops/init.hpp>
#include <poprithms/common/compute/ops/unaryelementwise.hpp>
#include <poprithms/common/compute/ops/withcallees.hpp>
#include <poprithms/common/compute/prune/pruner.hpp>
#include <poprithms/common/compute/slickgraph.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/program/prune/prune.hpp>

namespace {

void assertWithError(bool b, const std::string &ctxt = {}) {
  if (!b) {
    throw poprithms::test::error("Test with assertWithError failed. " + ctxt);
  }
}

using namespace poprithms::common::compute;
enum class Prune0Test { Test22, Test31 };
void testCustomPrune0(Prune0Test pruneTestType) {

  SlickGraph m;

  // Sub-graph with 3 inputs and 3 outputs.
  auto sg0 = m.createSubGraph("sg0");

  // The 3 scalar inputs of sg0:
  auto ins = sg0.variables(DType::Int32, {{}, {}, {}}, m.host());

  // The 3 scalar outputs of sg0:
  auto out0 = ins[0] + ins[1];
  auto out1 = ins[1] + ins[2];
  auto out2 = ins[2] + ins[0];

  const int64_t cnt{3};
  auto sg1  = m.createSubGraph("sg1");
  auto ins1 = sg1.variables(DType::Int32, {{cnt}, {cnt}, {cnt}}, m.host());

  std::vector<std::pair<TensorId, TensorId>> stackedIns{
      {ins1[0], ins[0]}, {ins1[1], ins[1]}, {ins1[2], ins[2]}};

  // Repeat has 2 outputs, both unpruneable.
  if (pruneTestType == Prune0Test::Test22) {

    std::vector<std::pair<TensorId, IsStackedCopy>> rptOuts{
        {out0, IsStackedCopy::Yes}, {out2, IsStackedCopy::Yes}};

    const auto rpt = sg1.repeat(sg0, cnt, stackedIns, {}, rptOuts);

    // both of the repeat outputs lead to the unpruneable.
    auto loss = out0.dstInCaller(rpt) + out2.dstInCaller(rpt);

    m.setRunnable({sg1});

    const auto nPrePrune = m.nOps();
    Pruner::prune(m, {loss});
    if (nPrePrune - m.nOps() != 1 || m.isLive(out1.opId())) {
      throw poprithms::test::error("Expected just the 1 add to be removed");
    }
  }

  // Repeat has 3 outputs, but only 1 is unpruneable.
  if (pruneTestType == Prune0Test::Test31) {

    std::vector<std::pair<TensorId, IsStackedCopy>> rptOuts{
        {out0, IsStackedCopy::Yes},
        {out1, IsStackedCopy::Yes},
        {out2, IsStackedCopy::Yes}};
    const auto rpt = sg1.repeat(sg0, cnt, stackedIns, {}, rptOuts);

    // both of the repeat outputs lead to the unpruneable.
    auto loss = out1.dstInCaller(rpt);

    m.setRunnable({sg1});
    Pruner::prune(m, {loss});

    assertWithError(m.nInTensors(rpt) == 2);
    assertWithError(m.nOutTensors(rpt) == 1);
    assertWithError(m.isLive(out1.opId()));
    assertWithError(!m.isLive(out0.opId()));
    assertWithError(!m.isLive(out2.opId()));
  }
}

void testCustomPrune1() {

  SlickGraph m;
  //
  //
  //  in0 ---> out0  -------> to unpruneable.
  //   ^
  //   |   carry
  //   +------<-------+
  //                  |
  //                  ^
  //  in1 ---------> out1
  //
  //  in2 --------------> out2
  //
  //
  //
  auto sg0  = m.createSubGraph("sg0");
  auto ins  = sg0.variables(DType::Int32, {{}, {}, {}}, m.host());
  auto out0 = ins[0].copy();
  auto out1 = ins[1].relu().sin();
  auto out2 = ins[2].copy();

  const int64_t cnt{3};
  auto sg1 = m.createSubGraph("sg1");

  // in1[0] is not stacked, ins1[1] and ins1[2] are:
  auto ins1 = sg1.variables(DType::Int32, {{}, {cnt}, {cnt}}, m.host());

  const auto rpt = sg1.repeat(sg0,
                              cnt,
                              // stacked:
                              {{ins1[1], ins[1]}, {ins1[2], ins[2]}},
                              // carried:
                              {{{ins1[0], ins[0], out1}}},
                              {{out0.id(), IsStackedCopy::Yes}});

  Tensor loss = out0.dstInCaller(rpt);
  m.setRunnable({sg1});
  Pruner::prune(m, {loss});
  assertWithError(
      m.isLive(out1.opId()),
      "out1 is carried back to in0, which is on a path to the loss, so "
      "out1 cannot be pruned");
  assertWithError(m.isLive(out0.opId()));
  assertWithError(!m.isLive(out2.opId()));

  assertWithError(
      m.nOutTensors(rpt) == 1,
      "The repeat op should only have 1 output, the one which is a path "
      "to the unpruneable tensor");

  assertWithError(!m.isLive(ins[2].opId()));
}

} // namespace

int main() {

  // prune with repeat op:
  testCustomPrune0(Prune0Test::Test22);
  testCustomPrune0(Prune0Test::Test31);
  testCustomPrune1();

  return 0;
}
