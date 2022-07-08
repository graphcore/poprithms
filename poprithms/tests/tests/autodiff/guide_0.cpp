// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <poprithms/autodiff/guide/guide.hpp>
#include <poprithms/autodiff/guide/objective.hpp>
#include <poprithms/autodiff/guide/traversals.hpp>
#include <poprithms/autodiff/testutil/testgraphinfo.hpp>
#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/util/printiter.hpp>

namespace {
using namespace poprithms;
using namespace poprithms::autodiff;
using poprithms::autodiff::testutil::TestGraphInfo;

using poprithms::common::multiout::InIndices;
using poprithms::common::multiout::OutIndices;

void baseTest(const TestGraphInfo &testGraph,
              const autodiff::guide::Objective &objective,
              OpIds expectedToRerun,
              TensorIds expectedWithGrads) {

  guide::Guide guide_(objective, testGraph);

  auto toRerun = guide_.opsToRerun();
  std::sort(toRerun.begin(), toRerun.end());
  std::sort(expectedToRerun.begin(), expectedToRerun.end());

  if (toRerun != expectedToRerun) {
    std::ostringstream oss;
    oss << "Incorrect set of ops to rerun. Expected ";
    poprithms::util::append(oss, expectedToRerun);
    oss << " but observed ";
    poprithms::util::append(oss, toRerun);
    oss << ". This for Guide " << guide_ << " and guide::Objective "
        << objective << ". ";
    throw poprithms::test::error(oss.str());
  }

  auto withGrads_ = guide_.nonGradsWithGrads();
  auto withGrads  = TensorIds(withGrads_.cbegin(), withGrads_.cend());
  std::sort(expectedWithGrads.begin(), expectedWithGrads.end());
  if (withGrads != expectedWithGrads) {
    std::ostringstream oss;
    oss << "Incorrect set of tensors with gradients. Expected "
        << expectedWithGrads << " but observed " << withGrads
        << ". This for Guide " << guide_ << " and guide::Objective "
        << objective << ". ";
    throw poprithms::test::error(oss.str());
  }
}

void test0() {

  // simple chain of ops:
  // 0 -> 1 -> 2.
  //
  //
  TestGraphInfo testGraph;
  testGraph.insertNoFlow({}, "op0");

  testGraph.insert({/* ins required (op id, output index) */ {{0, 0}},
                    /* number of outputs = */ 1,
                    /* inputs required = */ {},
                    /* outputs required = */ {0},
                    /* flows */ {{0, 0}},
                    "op1"});

  testGraph.insert({/* ins required (op id, output index) */ {{1, 0}},
                    /* number of outputs = */ 1,
                    /* inputs required = */ {},
                    /* outputs required = */ {0},
                    /* flows */ {{0, 0}},
                    "op2"});

  //    checkpoint
  //       .
  //       .
  //       0 -----------> 1 ------------> 2.
  //       .                              .
  //       .                              .
  //   target                        input grad
  //
  auto objective = guide::Objective::outOfGraph({{2, 0}}, // grads provided
                                                {{0, 0}}, // checkpoints
                                                {{0, 0}}  // targets
  );

  // we must compute the gradient of {1,0}, and then the gradient of {0,0}
  // (the target).
  //
  // To compute the gradient of {1,0} we must back-prop through op 2, as op
  // 2 is the consumer of {1,0}.
  //
  // to back-prop through op 2, we need the output of op 2 ({2,0}) as per
  // the requirements specified (see ££). but {2,0} is not a checkpoint, so
  // op 2 must be rerun.
  //
  // To compute the gradient of {0,0} we must back-peop through op 1. By the
  // same argument above, we must rerun op 1. Thus:
  OpIds expectedReruns{1, 2};
  TensorIds expectedWithGrads{{0, 0}, {1, 0}, {2, 0}};
  baseTest(testGraph, objective, expectedReruns, expectedWithGrads);
}

void test1() {

  //
  // 0.0   0.1   0.2
  //  |           |
  // 1@0         2@0
  //
  //         -- no flow --
  //
  // 1.0         2.0
  //  |           |
  // 3@0         3@1
  //
  //       3.0 :   the output with a gradient provided.

  TestGraphInfo testGraph;
  testGraph.insert({{}, 3, {}, {}, {}, "op0"});
  testGraph.insert({{{0, 0}}, 1, {}, {0}, {{0, 0}}, "op1"});
  testGraph.insert({{{0, 2}}, 1, {}, {}, {}, "op2"});
  testGraph.insert({{{1, 0}, {2, 0}}, 1, {}, {0}, {{0, 0}, {0, InIndex(1)}}});

  {
    auto objective =
        guide::Objective::outOfGraph({{3, 0}},                 // grads in
                                     {{0, 0}, {0, 1}, {0, 2}}, // checkpoints
                                     {{0, 0}}                  // targets
        );
    // must rerun 3 to get 3.0, to get grad of 3@0.
    // must return 1 and 2 get get inputs to 3.
    // 0's outputs are all checkpointed.
    OpIds expectedReruns({1, 2, 3});
    TensorIds expectedWithGrads({{0, 0}, {1, 0}, {3, 0}});
    baseTest(testGraph, objective, expectedReruns, expectedWithGrads);
  }

  // what if not all of 0's outputs are checkpointed?
  {

    bool caught = false;
    try {
      auto objective =
          guide::Objective::outOfGraph({{3, 0}},         // grads in
                                       {{0, 0}, {0, 1}}, // checkpoints
                                       {{0, 0}}          // targets
          );
      // must rerun 3 to get 3.0, to get grad of 3@0.
      // must return 1 and 2 get get inputs to 3.
      // 0's outputs are all checkpointed.
      OpIds expectedReruns({0, 1, 2, 3});
      TensorIds expectedWithGrads({{0, 0}, {1, 0}, {3, 0}});
      baseTest(testGraph, objective, expectedReruns, expectedWithGrads);
    } catch (const poprithms::error::error &) {
      caught = true;
    }
    if (!caught) {
      throw poprithms::test::error(
          "without {0,2} checkpointed, 0 needs to be rerun, so that 2 can "
          "be "
          "rerun. but 0 has no inputs, and thus in this test class is not "
          "rerunnable (assumed to be a variable initialization op). ");
    }
  }

  // what if we want gradients for all the outputs of 0?
  //
  // That's fine. The other tensors will just have zero gradients.
  {
    auto objective =
        guide::Objective::outOfGraph({{3, 0}},                 // grads in
                                     {{0, 0}, {0, 1}, {0, 2}}, // checkpoints
                                     {{0, 0}, {0, 1}, {0, 2}}  // targets
        );
    // must rerun 3 to get 3.0, to get grad of 3@0.
    // must return 1 and 2 get get inputs to 3.
    // 0's outputs are all checkpointed.
    OpIds expectedReruns({1, 2, 3});
    TensorIds expectedWithGrads({{0, 0}, {0, 1}, {0, 2}, {1, 0}, {3, 0}});
    baseTest(testGraph, objective, expectedReruns, expectedWithGrads);
  }
}

void test2() {

  //
  //       target
  //         .
  //         .
  //   0 -> 0.0 -> 1 -> 1.0 --> 2
  //                 -> 1.1 --> 3  ... gradient in here

  TestGraphInfo testGraph;
  // using machine::mockcontroler::Op;
  testGraph.insertNoFlow({}, "op0");
  testGraph.insert(
      {{{0, 0}}, 2, {}, {0, 1}, {{OutIndex(1), 0}, {0, 0}}, "op1"});
  testGraph.insert({{{1, 0}}, 1, {}, {0}, {{0, 0}}, "op2"});
  testGraph.insert({{{1, 1}}, 1, {}, {0}, {{0, 0}}, "op3"});

  {
    auto objective = guide::Objective::outOfGraph({{3, 0}}, // grads in
                                                  {{0, 0}}, // checkpoints
                                                  {{0, 0}}  // targets
    );

    // what's perhaps unexpected is that we expect a gradient for 1,1.
    // This is required though, as {0,1,1} is a gradient flowing traversal.
    // This tensor will be zero.
    baseTest(testGraph, objective, {1, 3}, {{0, 0}, {1, 0}, {1, 1}, {3, 0}});
  }
}

void test3() {

  //                         grad in
  //   target                  .
  //     .                     .
  //     .                     .
  //     0 --->-+- 2 -+- 4 -+- 6
  //            v     v     v
  //            |     |     |
  //            v     ^     ^
  //     1 --->-+- 3 -+- 5 -+- 7
  //     .                     .
  //     .                     .
  //   target                  .
  //                          grad in
  //
  //
  //
  // we will test that checkpoints are most effetive if the form a clean cut
  // of the graph.
  //

  TestGraphInfo testGraph;

  // no inputs to op0 and op1:
  testGraph.insertNoFlow({}, "op0");
  testGraph.insertNoFlow({}, "op1");
  //      inputs      n-outs
  //         |           |  required inputs for autodiff
  //         |           |  |   required outputs for autodiff
  //         |           |  |   |        gradient flows
  //         |           |  |   |              |               name
  //         |           |  |   |              |                 |
  //   ===============  ==  ==  ==   ========================   =====
  testGraph.insert(
      {{{0, 0}, {1, 0}}, 1, {}, {}, {{0, InIndex(1)}, {0, 0}}, "op2"});
  testGraph.insert(
      {{{0, 0}, {1, 0}}, 1, {}, {}, {{0, InIndex(1)}, {0, 0}}, "op3"});
  testGraph.insert(
      {{{2, 0}, {3, 0}}, 1, {}, {}, {{0, InIndex(1)}, {0, 0}}, "op4"});
  testGraph.insert(
      {{{2, 0}, {3, 0}}, 1, {}, {}, {{0, InIndex(1)}, {0, 0}}, "op5"});
  testGraph.insert(
      {{{4, 0}, {5, 0}}, 1, {}, {0}, {{0, InIndex(1)}, {0, 0}}, "op6"});
  testGraph.insert(
      {{{4, 0}, {5, 0}}, 1, {}, {0}, {{0, InIndex(1)}, {0, 0}}, "op7"});

  // Id Ins             nOut insRequired outsRequired flows       name
  // -- ---             ---- ----------- ------------ -----       ----
  // 0  ()              1    ()          ()           ()          op0
  // 1  ()              1    ()          ()           ()          op1
  // 2  ((op=0),(op=1)) 1    ()          ()           (1<-0,0<-0) op2
  // 3  ((op=0),(op=1)) 1    ()          ()           (1<-0,0<-0) op3
  // 4  ((op=2),(op=3)) 1    ()          ()           (1<-0,0<-0) op4
  // 5  ((op=2),(op=3)) 1    ()          ()           (1<-0,0<-0) op5
  // 6  ((op=4),(op=5)) 1    ()          (0)          (1<-0,0<-0) op6
  // 7  ((op=4),(op=5)) 1    ()          (0)          (1<-0,0<-0) op7

  {
    auto objective =
        guide::Objective::outOfGraph({{6, 0}, {7, 0}}, // grads in
                                     {{0, 0}, {1, 0}}, // checkpoints
                                     {{0, 0}, {1, 0}}  // targets
        );
    TensorIds expectedGrads;
    for (uint64_t i = 0; i < 8; ++i) {
      expectedGrads.push_back(TensorId(OpId(i), OutIndex(0)));
    }
    baseTest(testGraph, objective, {2, 3, 4, 5, 6, 7}, expectedGrads);
  }

  // checkpoints don't form a clean "cut" so still need all the
  // uncheckpointed ops to be recomputed:
  {
    auto objective = guide::Objective::outOfGraph(
        {{6, 0}, {7, 0}},                         // grads in
        {{0, 0}, {1, 0}, {2, 0}, {5, 0}, {6, 0}}, // checkpoints
        {{0, 0}, {1, 0}}                          // targets
    );
    TensorIds expectedGrads;
    for (uint64_t i = 0; i < 8; ++i) {
      expectedGrads.push_back(TensorId(OpId(i), OutIndex(0)));
    }
    baseTest(testGraph, objective, {3, 4, 7}, expectedGrads);
  }

  // checkpoints form a clean cut:
  {
    auto objective =
        guide::Objective::outOfGraph({{6, 0}, {7, 0}},         // grads in
                                     {{0, 0}, {4, 0}, {5, 0}}, // checkpoints
                                     {{0, 0}, {1, 0}}          // targets
        );
    TensorIds expectedGrads;
    for (uint64_t i = 0; i < 8; ++i) {
      expectedGrads.push_back(TensorId(OpId(i), OutIndex(0)));
    }
    baseTest(testGraph, objective, {6, 7}, expectedGrads);
  }
}

void testTraversals0() {

  TestGraphInfo testGraph;
  // arguments are:
  //     inputs
  //     |  n-outs
  //     |  |  required inputs for autodiff
  //     |  |  |  required outputs for autodiff
  //     |  |  |  |  gradient flows
  //     |  |  |  |  |  name
  //     |  |  |  |  |  |
  auto x0 = testGraph.insert({{}, 1, {}, {}, {}, "x0"});
  auto x1 = testGraph.insert({{}, 1, {}, {}, {}, "x1"});

  // 2 inputs, 2 outputs, all possible gradient flows
  auto fullFlow = [&testGraph](TensorId in0, TensorId in1) -> OpId {
    return testGraph.insert(
        {{in0, in1},
         2,
         {},
         {},
         {{OutIndex(0), InIndex(1)}, {0, 0}, {1, 0}, {1, 1}},
         "x2"});
  };

  //
  //  x0 --+   +== x2 --+
  //       +===+        +--x4
  //  x1 --+   +== x3 --+
  //

  auto x2 = fullFlow({x0, 0}, {x1, 0});
  auto x3 = fullFlow({x0, 0}, {x1, 0});
  auto x4 = fullFlow({x2, 0}, {x3, 1});

  auto dx4_0 = testGraph.insert({{}, 1, {}, {}, {}, "x0"});

  auto objective =
      guide::Objective::inGraph({{x4, 0}},          // grads in for
                                {{x0, 0}, {x1, 0}}, // checkpoints
                                {{x0, 0}},          // target
                                {{dx4_0, 0}});      // grad in

  guide::Traversals travs(objective, testGraph);

  // The traversals:
  // ((in=0,op=2,out=0),
  //  (in=0,op=3,out=1),
  //  (in=0,op=4,out=0),
  //  (in=1,op=4,out=0))

  if (travs.inIndicesTraversed(x4) != InIndices{0, 1}) {
    throw poprithms::test::error(
        "x4's 0'th output is the 'loss'. Both of x4's inputs effect x4's "
        "0'th output, and are on a path from the target of differentiation.");
  }

  if (travs.outIndicesTraversed(x4) != OutIndices{0}) {
    throw poprithms::test::error("x4's 0'th output is the 'loss', and the "
                                 "it's 1'st output leads nowhere");
  }

  if (travs.outIndicesTraversed(x3) != OutIndices{1} ||
      travs.inIndicesTraversed(x3) != InIndices{0}) {
    throw poprithms::test::error("x3 is traversed on exacly 1 path from "
                                 "target to loss: input 0 to output 1");
  }

  if (travs.outIndicesTraversed(x2) != OutIndices{0} ||
      travs.inIndicesTraversed(x2) != InIndices{0}) {
    throw poprithms::test::error("x3 is traversed on exacly 1 path from "
                                 "target to loss: input 0 to output 0");
  }
}

} // namespace

int main() {
  test0();
  test1();
  test2();
  test3();
  testTraversals0();
  return 0;
}
