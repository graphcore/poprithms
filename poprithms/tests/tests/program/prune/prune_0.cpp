// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <set>
#include <sstream>

#include <testutil/program/callstack/graph.hpp>
#include <testutil/program/callstack/querier.hpp>

#include <poprithms/common/schedulable/graph.hpp>
#include <poprithms/common/schedulable/op.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/program/prune/prune.hpp>

namespace {

using namespace poprithms::program::prune;
using namespace poprithms::program;
using namespace poprithms::program::callstack_test;

class TestMutator final : public prune::Mutator {
private:
  Graph &g_;

public:
  TestMutator(Graph &g) : g_(g) {}
  void removeInputs(OpId opId, const InIndices &ins) final {
    g_.removeInputs(opId, ins);
  }

  void removeOutputs(OpId opId, const OutIndices &outIndices) final {
    g_.removeOutputs(opId, outIndices, OptionalTensorIds(outIndices.size()));
  }

  void removeOp(OpId opId, const std::string &ctxt) final {
    for (OutIndex o = 0; o < g_.nOutTensors(opId); ++o) {
      if (g_.nConsumptionIds({opId, o}) != 0) {
        throw poprithms::test::error(
            "can't remove op with consumers when pruning");
      }
    }

    OptionalTensorIds opts(g_.op(opId).nOutTensors());
    g_.removeOp(opId, opts, ctxt);
  }
};

void verify(
    const Graph &g,
    const OpIds &expectedRemoved,
    const std::vector<std::pair<OpId, TensorIds>> &expectedInDests,
    const std::vector<std::pair<OpId, CopyOuts>> &expectedOutSources) {
  g.verifyValid();

  for (auto opId : g.opIdsAllSubGraphs()) {
    if (std::find(expectedRemoved.cbegin(), expectedRemoved.cend(), opId) !=
        expectedRemoved.cend()) {
      std::ostringstream oss;
      oss << opId << " was not removed, but is should have been.";
      throw poprithms::test::error(oss.str());
    }
  }
  for (auto re : g.removalEvents().events) {
    if (std::find(expectedRemoved.cbegin(),
                  expectedRemoved.cend(),
                  re.opId) == expectedRemoved.cend()) {
      std::ostringstream oss;
      oss << re.opId << " was removed, but is should not have been.";
      throw poprithms::test::error(oss.str());
    }
  }
  for (const auto &expectedIn : expectedInDests) {
    auto inCopies = g.op(expectedIn.first).inCopies();
    if (inCopies.dstIds() != expectedIn.second) {
      std::ostringstream oss;
      oss << "Incorrect inputs. "
          << "Expected " << expectedIn.second << " but observed " << inCopies;
      throw poprithms::test::error(oss.str());
    }
  }

  for (const auto &expectedOut : expectedOutSources) {
    auto outCopies = g.op(expectedOut.first).outCopies();
    if (outCopies != expectedOut.second) {
      std::ostringstream oss;
      oss << "Incorrect outputs. "
          << "Expected " << expectedOut.second << " but observed "
          << outCopies;
      throw poprithms::test::error(oss.str());
    }
  }
}

void testCall0() {

  Graph g;

  /**
   *  in0 -+
   *       +--- add (used)
   *       |
   *       +--- sub (unused)
   *  in1 -+
   *
   * */
  auto sg0       = g.createSubGraphId("sg0");
  auto in0       = TensorId(g.insert({}, 1, sg0, "in0"), 0);
  auto in1       = TensorId(g.insert({}, 1, sg0, "in1"), 0);
  auto usedOut   = TensorId(g.insert({in0, in1}, 1, sg0, "add"), 0);
  auto unusedOut = TensorId(g.insert({in0, in1}, 1, sg0, "sub"), 0);

  /** z0 = sg0(x0, x1). */
  auto sg1 = g.createSubGraphId("sg1");
  auto x0  = TensorId(g.insert({}, 1, sg1, "x0"), 0);
  auto x1  = TensorId(g.insert({}, 1, sg1, "x1"), 0);

  CopyOuts outs({/* OutIndex = 0*/ {usedOut}, /* OutIndex = 1*/ {unusedOut}});

  auto z0 = g.insert(sg1,
                     {sg0},
                     CopyIns({{x0, in0, 0}, {x1, in1, 0}}),
                     outs,
                     {},
                     {},
                     "call");
  (void)z0;

  SubGraphIds callables({sg1});

  // do not prune the out destination of "usedOut":
  TensorIds backSources({{z0, 0}});
  TestMutator mut(g);
  prune::Pruner::prune(Querier(g), mut, callables, backSources);

  verify(g,
         /* The ops which we expect to removed: */ {unusedOut.opId()},
         /* The call inputs which we expect to remain: */ {{z0, {in0, in1}}},
         /* The call outputs which we expect to remain: */
         {{z0, {{{usedOut}}}}});

  std::cout << g << std::endl;
}

void testSwitch0() {

  for (int testCase = 0; testCase < 4; ++testCase) {
    std::cout << "testCase = " << testCase << std::endl;

    Graph g;

    /**
     *  in00  --> sqrt --> out00
     *  in01  --> cbrt --> out01.
     * */
    auto sg0   = g.createSubGraphId("sg0");
    auto in00  = TensorId(g.insert({}, 1, sg0, "in00"), 0);
    auto out00 = TensorId(g.insert({in00}, 1, sg0, "sqrt"), 0);
    auto in01  = TensorId(g.insert({}, 1, sg0, "in0"), 0);
    auto out01 = TensorId(g.insert({in01}, 1, sg0, "cbrt"), 0);

    /**
     *       +--> relu --> out10
     *  in1 -+
     *       +--> abs ---> out11
     * */
    auto sg1   = g.createSubGraphId("sg1");
    auto in1   = TensorId(g.insert({}, 1, sg1, "in1"), 0);
    auto out10 = TensorId(g.insert({in1}, 1, sg1, "relu"), 0);
    auto out11 = TensorId(g.insert({in1}, 1, sg1, "abs"), 0);

    /**
     * switch between sg0 and sg1. A single input to switch will get copied to
     * in00 and in01 if sg0 is run, and to in1 if sg1 is run. The decision of
     * which sub-graph to run is controlled by a switch condition tensor
     * #cond.
     *
     * Copy outs: there are 2 output indices.
     *
     *  if sg0 is run:
     *     out00 is output at index 0 and nothing is output at index 1.
     *
     *  if sg1 is run:
     *     out10 is output at index 0 and out11 is output at index 1.
     * */
    auto sg2  = g.createSubGraphId("sg2");
    auto in2  = TensorId(g.insert({}, 1, sg2, "in2"), 0);
    auto cond = TensorId(g.insert({}, 1, sg2, "cond"), 0);

    auto outs = CopyOuts::fromOptionals(
        {{out00, out10}, {{OptionalTensorId{}, out11}}});
    auto sw = g.insert(sg2,
                       {sg0, sg1},
                       {{{in2, in00, 0}, {in2, in01, 0}, {in2, in1, 1}}},
                       outs,
                       cond,
                       {},
                       "switch");

    // output is sum of 2 switch outputs. This is slightly unrealistic, as
    // you'd never unconditionally use output at index 1, as sg0 does not
    // output a tensor at this index. Good enough for testing though.
    auto samba = TensorId(g.insert({{sw, 0}, {sw, 1}}, 1, sg2, "sum"), 0);

    TestMutator mut(g);
    Querier querier(g);

    if (testCase == 0) {

      SubGraphIds callables{sg2};
      TensorIds backSources({samba});
      prune::Pruner::prune(querier, mut, callables, backSources);

      verify(g,
             /* The ops which we expect to removed: */
             {in01.opId(), out01.opId()},
             /* The switch inputs which we expect to remain: */
             {{sw, {in00, in1}}},
             /* The switch outputs which we expect to remain: */
             {{sw, outs}});
    }

    if (testCase == 1) {

      SubGraphIds callables{sg2};
      TensorIds backSources({{sw, 0}});
      prune::Pruner::prune(querier, mut, callables, backSources);
      verify(g,
             /* The ops which we expect to removed: */
             {samba.opId(), in01.opId(), out01.opId(), out11.opId()},
             /* The switch inputs which we expect to remain: */
             {{sw, {in00, in1}}},
             /* The switch outputs which we expect to remain: */
             {{sw, {{{out00, out10}}}}});
    }

    if (testCase == 2) {

      SubGraphIds callables{sg2};
      TensorIds backSources({{sw, 1}});
      prune::Pruner::prune(querier, mut, callables, backSources);
      verify(g,
             /* The ops which we expect to removed: */
             {samba.opId(),
              in00.opId(),
              in01.opId(),
              out00.opId(),
              out01.opId(),
              out10.opId()},
             /* The switch inputs which we expect to remain: */
             {{sw, {in1}}},
             /* The switch outputs which we expect to remain: */
             {{sw, CopyOuts::fromOptionals({{OptionalTensorId{}, out11}})}});
    }

    if (testCase == 3) {

      // Actually prune the whole switch op, user just want to run the
      // sub-graphs individually.
      SubGraphIds callables{sg0, sg1};
      TensorIds backSources({out00, out10});
      prune::Pruner::prune(querier, mut, callables, backSources);
      g.verifyValid();
      if (g.nOps() != 4) {
        throw poprithms::test::error(
            "Expected just ins and outs of sqrt and relu to be left");
      }
    }
  }
}
void testRepeat0() {

  Graph g;
  auto sg0        = g.createSubGraphId("sg0");
  auto in0        = TensorId(g.insert({}, 1, sg0, "in0"), 0);
  auto in1        = TensorId(g.insert({}, 1, sg0, "in1"), 0);
  auto sum        = TensorId(g.insert({in0, in1}, 1, sg0, "sum"), 0);
  auto sumSquared = TensorId(g.insert({sum}, 1, sg0, "sumSquared"), 0);

  auto sg1 = g.createSubGraphId("sg1");
  auto in2 = TensorId(g.insert({}, 1, sg1, "in2"), 0);
  auto in3 = TensorId(g.insert({}, 1, sg1, "in3"), 0);
  auto rpt = g.insert(sg1,
                      {sg0},
                      {{{in2, in0, 0}, {in3, in1, 0}}},
                      /** outs = */ {{{in1}}},
                      {},
                      {{sum, in1}},
                      "rpt");

  // we don't expect sum to be pruned, as it the copy-back source of in1.
  // sumSquared hoever is on a road to nowhere with no communication back.

  SubGraphIds callables({sg1});

  // do not prune the out destination of "usedOut":
  TensorIds backSources({{rpt, 0}});
  TestMutator mut(g);
  prune::Pruner::prune(Querier(g), mut, callables, backSources);

  verify(g,
         /* The ops which we expect to removed: */
         {sumSquared.opId()},
         /* The repeat inputs which we expect to remain: */
         {{rpt, {in0, in1}}},
         /* The repeat outputs which we expect to remain: */
         {{rpt, {{{in1}}}}});
}
} // namespace

int main() {
  testCall0();
  testSwitch0();
  testRepeat0();
  return 0;
}
