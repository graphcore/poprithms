// Copyright 2021 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <iostream>
#include <sstream>

#include <testutil/autodiff/testgraphmutator.hpp>
#include <testutil/schedule/shift/shiftcommandlineoptions.hpp>

#include <poprithms/autodiff/core/autodiff.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms;
using namespace poprithms::autodiff;
using namespace poprithms::autodiff::core;
using poprithms::autodiff::test::Op;
using poprithms::autodiff::test::TestGraphInfo;
using poprithms::autodiff::test::TestGraphMutator;

// A test helper class which attempts to find an op meeting a set of
// conditions in the graph.
//
// ins:                  the inputs to the op.
// orderOfInsMatters:    does the order of the inputs matter?
// t:                    the type of the op
// name (optional)       the name of the op. if empty, it is ignored.

class Getter {

private:
  const TestGraphInfo &testGraph;

public:
  Getter(const TestGraphInfo &c_) : testGraph(c_) {}

  OpId operator()(const TensorIds &ins,
                  bool orderOfInsMatters,
                  Op::Type t,
                  const std::string &name) {
    // for comparing 2 vectors for which the order does not matter:
    auto ordered = [](auto x) {
      std::sort(x.begin(), x.end());
      return x;
    };

    for (uint64_t i = 0; i < testGraph.nOps(); ++i) {
      const auto &op_ = testGraph.op(i);
      if (op_.type == t) {
        if (name.empty() || name == op_.name) {
          if (orderOfInsMatters && op_.ins == ins) {
            return i;
          }
          if (!orderOfInsMatters && ordered(op_.ins) == ordered(ins)) {
            return i;
          }
        }
      }
    }

    std::ostringstream oss;
    oss << "Failed to retrieve an op which specified the query. \n   ins="
        << ins << "\n   orderOfInsMatters=" << orderOfInsMatters
        << "\n   Op::Type=" << Op::str(t) << "\n   name=\"" << name << "\"";

    throw poprithms::test::error(oss.str());
  }

  // verify that no ops have "frag" in their name.
  void assertNone(const std::string &frag) {
    for (uint64_t i = 0; i < testGraph.nOps(); ++i) {
      auto x = testGraph.op(i).name;
      if (x.find(frag) != std::string::npos) {
        std::ostringstream oss;
        oss << "Failed in Getter::assertNone(frag = \"" << frag << "\"). "
            << "This sub-string was found in op #" << i
            << ", whose name is \"" << x << "\".";
        throw poprithms::test::error(oss.str());
      }
    }
  }

  void assertCount(Op::Type t, uint64_t n) {
    OpIds matches;
    for (uint64_t i = 0; i < testGraph.nOps(); ++i) {
      if (testGraph.op(i).type == t) {
        matches.push_back(OpId(i));
      }
    }

    if (matches.size() != n) {
      std::ostringstream oss;
      oss << "Failure in assertCount. "
          << "Expected " << n << " of type " << t << ", but observed "
          << matches.size() << ". "
          << "The matching ops were ";
      poprithms::util::append(oss, matches);
      oss << ". ";
      throw poprithms::test::error(oss.str());
    }
  }
};

void testMatmul0() {
  //
  // Test of basic matmul-like operation:
  //
  //         v0      v1
  //         |       |
  //       [v0,0] [v1,0]
  //         |       |
  //         +--+----+
  //            |
  //           (mm0)
  //            |
  //         [mm0,0] .... gradIn
  //

  TestGraphInfo testGraph = []() {
    TestGraphInfo testGraph_;
    auto v0  = testGraph_.insertNoFlow({}, "v0", Op::Type::Variable);
    auto v1  = testGraph_.insertNoFlow({}, "v1", Op::Type::Variable);
    auto mm0 = testGraph_.matmul(v0, v1, "mm0");
    const auto g =
        guide::Objective::outOfGraph({{mm0, 0}}, {v0, v1}, {v0, v1});
    TestGraphMutator a(testGraph_);
    Autodiff(g, testGraph_, a);
    return testGraph_;
  }();

  std::cout << testGraph << std::endl;
  //  Id Type       Ins                    nOut  name
  //  -- ----       ---                    ----  ----
  //  0  Unknown    ()                     1     v0
  //  1  Unknown    ()                     1     v1
  //  2  Matmul     ((op=0),(op=1))        1     mm0
  //  3  Variable   ()                     1     checkpoint/(op=0)
  //  4  Variable   ()                     1     checkpoint/(op=1)
  //  5  Zero       ((op=0))               1     init-grad-of/(op=0)
  //  6  Zero       ((op=1))               1     init-grad-of/(op=1)
  //  7  Zero       ((op=2))               1     init-grad-of/(op=2)
  //  8  Variable   ()                     1     grad-in-of/(op=2)
  //  9  Add        ((op=7),(op=8))        1     Add
  //  10 MatmulGrad ((op=9),(op=3),(op=4)) 2     grad-of-op-2-inputs-(0,1)
  //  11 Add        ((op=5),(op=10))       1     Add
  //  12 Add        ((op=6),(op=10,out=1)) 1     Add

  Getter get0(testGraph);

  // 2 checkpoints:
  auto cp0 =
      get0({}, true, Op::Type::Variable, Autodiff::genCheckpointName({0, 0}));

  auto cp1 =
      get0({}, true, Op::Type::Variable, Autodiff::genCheckpointName({1, 0}));

  // 3 zeros,
  auto z0 = get0({}, true, Op::Type::Zero, Autodiff::genInitGradName({0, 0}));
  auto z1 = get0({}, true, Op::Type::Zero, Autodiff::genInitGradName({1, 0}));
  auto zmm =
      get0({}, true, Op::Type::Zero, Autodiff::genInitGradName({2, 0}));

  // 1 variable, "grad in of"
  auto gIn =
      get0({}, true, Op::Type::Variable, Autodiff::genInGradName({2, 0}));

  // 1 add, of gIn and zmm
  auto gmm = get0({{gIn, 0}, {zmm, 0}}, false, Op::Type::Add, "");

  // 1 matmul grad, with 3 inputs.
  gmm = get0({{gmm, 0}, {cp0, 0}, {cp1, 0}}, false, Op::Type::MatmulGrad, "");

  // 2 adds to get the final gradients.
  get0({{z0, 0}, {gmm, 0}}, false, Op::Type::Add, "");
  get0({{gmm, 1}, {z1, 0}}, false, Op::Type::Add, "");
}

// basic recompute.
void testRecompute0() {

  //
  //    v0  <- grad required. checkpointed.
  //    |
  //    x0
  //    |
  //    x1 <--- grad input
  //
  //
  //  Id Type        Ins               outsRequired flows    name
  //  -- ----        ---               ------------ -----    ----
  //  0  Variable    ()                ()           ()       v0
  //  1  Unknown     ((op=0))          (0)          (0<-0)   x0
  //  2  Unknown     ((op=1))          (0)          (0<-0)   x1
  //  3  Variable    ()                                      v0 checkpoint
  //  4  Unknown     ((op=3))                                x0 rerun
  //  5  Unknown     ((op=4))                                x1 rerun
  //  6  Zero        ()                                      ettestGraph.
  //  7  Zero        ()
  //  8  Zero        ()
  //  9  Variable    ()
  //  10 Add         ((op=8),(op=9))
  //  11 UnknownGrad ((op=10),(op=5))
  //  12 Add         ((op=7),(op=11))
  //  13 UnknownGrad ((op=12),(op=4))
  //  14 Add         ((op=6),(op=13))

  TestGraphInfo testGraph;
  auto v0 = testGraph.insertNoFlow({}, "v0", Op::Type::Variable);
  auto x0 =
      TensorId(testGraph.insert(Op({v0}, 1, {}, {0}, {{0, 0}}, "x0")), 0);
  auto x1 =
      TensorId(testGraph.insert(Op({x0}, 1, {}, {0}, {{0, 0}}, "x1")), 0);
  const auto g = guide::Objective::outOfGraph({x1}, {v0}, {v0});
  TestGraphMutator a(testGraph);
  Autodiff(g, testGraph, a);

  std::cout << testGraph << std::endl;
  Getter x(testGraph);

  // checkpoint:
  TensorId cp0(
      x({}, true, Op::Type::Variable, Autodiff::genCheckpointName(v0)), 0);

  // recomputed tensors:
  TensorId recomp0(
      x({cp0}, true, Op::Type::Unknown, Autodiff::genRerunName(x0.opId())),
      0);

  TensorId recomp1(x({recomp0},
                     true,
                     Op::Type::Unknown,
                     Autodiff::genRerunName(x1.opId())),
                   0);

  // initial (zero) grads
  TensorId zv0(x({}, true, Op::Type::Zero, Autodiff::genInitGradName(v0)), 0);
  TensorId zx0(x({}, true, Op::Type::Zero, Autodiff::genInitGradName(x0)), 0);
  TensorId zx1(x({}, true, Op::Type::Zero, Autodiff::genInitGradName(x1)), 0);

  // grad in:
  TensorId gIn(x({}, true, Op::Type::Variable, Autodiff::genInGradName(x1)),
               0);

  // gradient of x1:
  TensorId dx1(x({gIn, zx1}, false, Op::Type::Add, ""), 0);

  // gradient of x0:
  TensorId ingrad1(x({dx1, recomp1}, false, Op::Type::UnknownGrad, ""), 0);
  TensorId dx0(x({ingrad1, zx0}, false, Op::Type::Add, ""), 0);

  // gradient of the traget, v0:
  TensorId ingrad0(x({dx0, recomp0}, false, Op::Type::UnknownGrad, ""), 0);
  TensorId dv0(x({ingrad0, zv0}, false, Op::Type::Add, ""), 0);
}

// basic recompute, second test. In this test, the grad ops don't require
// any non-gradient tensors, and so we don't expect anything to be
// recomputed.
void testNoRecomputeWithAffine0() {

  // Something like
  //
  //  v0 -> x0 = scale(2.0) -> x1 = scale(3.0)
  //  where neither x0 nor x1 are needed to backpropagate.
  //
  TestGraphInfo testGraph;
  auto v0 = testGraph.insertNoFlow({}, "v0", Op::Type::Variable);
  TensorId x0(testGraph.insert(Op({v0}, 1, {}, {}, {{0, 0}}, "x0")), 0);
  TensorId x1(testGraph.insert(Op({x0}, 1, {}, {}, {{0, 0}}, "x1")), 0);
  const auto g = guide::Objective::outOfGraph({x1}, {v0}, {v0});
  TestGraphMutator a(testGraph);
  Autodiff(g, testGraph, a);
  Getter getter(testGraph);
  getter.assertNone(Autodiff::genRerunName(v0.opId()));
  getter.assertNone(Autodiff::genRerunName(x0.opId()));
  getter.assertNone(Autodiff::genCheckpointName(x0));
  getter.assertNone(Autodiff::genRerunName(x1.opId()));
}

void testNoFlow0() {
  TestGraphInfo testGraph;
  auto v0 = testGraph.insertNoFlow({}, "v0", Op::Type::Variable);
  auto x0 = testGraph.insert(
      Op({v0}, 2, {0}, {0, 1}, /* no gradient flows: */ {}, "x0"));
  const auto g =
      guide::Objective::outOfGraph({{x0, 0}}, {v0, {x0, 0}, {x0, 1}}, {v0});
  TestGraphMutator a(testGraph);
  Autodiff(g, testGraph, a);
  std::cout << testGraph << std::endl;
  Getter getter(testGraph);

  // clang-format off
  //
  //   Id Type     Ins             nOut insRequired outsRequired flows       name
  //   -- ----     ---             ---- ----------- ------------ -----       ----
  //   0  Variable ()              1    ()          ()           ()          v0
  //   1  Unknown  ((op=0))        2    (0)         (0,1)        ()          x0
  //   2  Variable ()              1    ()          ()           ()          checkpoint/(op=0)
  //   3  Variable ()              1    ()          ()           ()          checkpoint/(op=1)
  //   4  Variable ()              1    ()          ()           ()          checkpoint/(op=1,out=1)
  //   5  Zero     ()              1    ()          ()           ()          init-grad-of/(op=0)
  //   6  Zero     ()              1    ()          ()           ()          init-grad-of/(op=1)
  //   7  Variable ()              1    ()          ()           ()          grad-in-of/(op=1)
  //   8  Add      ((op=6),(op=7)) 1    ()          ()           (0<-1,0<-0) Add
  //
  // clang-format on
  //
  getter.assertCount(Op::Type::Add, 1);
  getter.assertCount(Op::Type::UnknownGrad, 0);

  // Still expect the machinery for the gradient of x0, a it's input
  // gradient is provided.
  getter.assertCount(Op::Type::Zero, 2);
}

void testComplexOp0() {

  std::cout << "\ntestComplexOp0\n" << std::endl;

  //      +---- flow ---------> .... < gradient in
  //      |
  // x0 --+---- flow   ------->  ... < no gradient in
  //      |
  //      +---- no flow   ----> ... < no gradient in

  TestGraphInfo testGraph;

  // op 0
  auto x0 = testGraph.insertNoFlow({}, "v0", Op::Type::Variable);

  // op1
  auto x1 = testGraph.insert(
      Op({x0}, 3, {}, {0, 1, 2}, {{OutIndex(1), 0}, {0, 0}}, "x1"));

  // op2
  auto x10 = testGraph.insert(Op({{x1, 0}}, 1, {}, {0}, {{0, 0}}, "x10"));

  // op3
  testGraph.insert(Op({{x1, 1}}, 1, {}, {0}, {{0, 0}}, "x11"));

  // op4
  testGraph.insert(Op({{x1, 2}}, 1, {}, {0}, {{0, 0}}, "x12"));
  const auto g = guide::Objective::outOfGraph({{x10, 0}}, {x0}, {x0});

  // clang-format off
  //  Id Type        Ins                                                nOut outsRequired flows       name
  //  -- ----        ---                                                ---- ------------ -----       ----
  //  0  Variable    ()                                                 1    ()           ()          v0
  //  1  Unknown     ((op=0))                                           3    (0,1,2)      (0<-1,0<-0) x1
  //  2  Unknown     ((op=1))                                           1    (0)          (0<-0)      x10
  //  3  Unknown     ((op=1,out=1))                                     1    (0)          (0<-0)      x11
  //  4  Unknown     ((op=1,out=2))                                     1    (0)          (0<-0)      x12
  //  5  Variable    ()                                                 1    ()           ()          checkpoint/(op=0)
  //  6  Unknown     ((op=5))                                           3    (0,1,2)      (0<-1,0<-0) rerun/1
  //  7  Unknown     ((op=6))                                           1    (0)          (0<-0)      rerun/2
  //  8  Zero        ()                                                 1    ()           ()          init-grad-of/(op=0)
  //  9  Zero        ()                                                 1    ()           ()          init-grad-of/(op=1)
  //  10 Zero        ()                                                 1    ()           ()          init-grad-of/(op=1,out=1)
  //  11 Zero        ()                                                 1    ()           ()          init-grad-of/(op=2)
  //  12 Variable    ()                                                 1    ()           ()          grad-in-of/(op=2)
  //  13 Add         ((op=11),(op=12))                                  1    ()           (0<-1,0<-0) Add
  //  14 UnknownGrad ((op=13),(op=7))                                   1    ()           ()          grad-of-op-2-inputs-(0)
  //  15 Add         ((op=9),(op=14))                                   1    ()           (0<-1,0<-0) Add
  //  16 UnknownGrad ((op=15),(op=10),(op=6),(op=6,out=1),(op=6,out=2)) 1    ()           ()          grad-of-op-1-inputs-(0)
  //  17 Add         ((op=8),(op=16))                                   1    ()           (0<-1,0<-0) Add
  //
  // clang-format on

  TestGraphMutator a(testGraph);
  Autodiff(g, testGraph, a);
  Getter getter(testGraph);

  // we expect exactly 4 initialization of zero ops, 1 for each of the
  // tensor which require a gradient. These are: 1) x0 : the target 2) op2,
  // which is where the input gradient flows from 3) op1 (the complex op) at
  // outputs #0 and #1. None for #2 as there is no flow from this output
  // index.
  getter.assertCount(Op::Type::Zero, 4);
  getter({}, true, Op::Type::Zero, Autodiff::genInitGradName(x0));
  getter({}, true, Op::Type::Zero, Autodiff::genInitGradName({x1, 0}));
  getter({}, true, Op::Type::Zero, Autodiff::genInitGradName({x1, 1}));
  getter({}, true, Op::Type::Zero, Autodiff::genInitGradName({x10, 0}));
}

void testComplexOp1() {

  // How gradients flow in thie example (lines within dotted squares).
  //
  //                       "multi" op
  //                  . . . . . . . . .
  //           +----0 .  <---+----    . -------+
  //           |      .      |        .        |
  //           +----1 .      +----    . ---+   |
  //           |      . . . . . . . . .    |   |
  // input  ---+                           v   v
  //           |                           0   1
  //           |       . . . . . . . . . . .  . . .
  //           +---> 2 . <---+             ^   ^  .
  //           |       .     |             |   |  .
  //           +---> 3 .     +-------------+---+  . ---> loss tensor
  //                   .                          .
  //                   . . . . . . . . . . .  . . .
  //                          "loss" op
  //
  TestGraphInfo testGraph;
  const auto input = testGraph.insertNoFlow({}, "input", Op::Type::Variable);

  const auto multi = testGraph.insert(Op({input, input},
                                         2,
                                         {},
                                         {0, 1},
                                         // no gradients flow to to input 1.
                                         {{OutIndex{0}, 0}, {1, 0}},
                                         "multi"));

  const auto loss =
      testGraph.insert(Op({input, input, {multi, 0}, {multi, 1}},
                          1,
                          {},
                          {0},
                          {{OutIndex(0), 0}, {0, 1}, {0, 2}},
                          "loss"));

  const auto g =
      guide::Objective::outOfGraph({{loss, 0}}, {input, {loss, 0}}, {input});
  TestGraphMutator a(testGraph);
  Autodiff(g, testGraph, a);
  Getter getter(testGraph);

  std::cout << testGraph << std::endl;

  // all 4 tensors have gradients.
  getter.assertCount(Op::Type::Zero, 4);

  // 5 adds.
  getter.assertCount(Op::Type::Add, 5);
}

// Like complexOp1, but
// 1) flows through complex are modified.
// 2) input order to multi changed.
void testComplexOp2() {

  TestGraphInfo testGraph;
  const auto input = testGraph.insertNoFlow({}, "input", Op::Type::Variable);

  const auto multi =
      testGraph.insert(Op({input, input},
                          2,
                          {},
                          {0, 1},
                          // new flows: all flows from output 0.
                          {{OutIndex{0}, 0}, {0, 1}},
                          "multi"));

  const auto loss =
      testGraph.insert(Op({{multi, 0}, {multi, 1}, input, input},
                          1,
                          {},
                          {0},
                          {{OutIndex(0), 0}, {0, 1}, {0, 2}},
                          "loss"));

  const auto g =
      guide::Objective::outOfGraph({{loss, 0}}, {input, {loss, 0}}, {input});
  TestGraphMutator a(testGraph);
  Autodiff(g, testGraph, a);
  Getter getter(testGraph);
  std::cout << testGraph << std::endl;

  // expect no initialized gradient for output 1 of multi.
  getter.assertCount(Op::Type::Zero, 3);

  // expect 5 adds.
  getter.assertCount(Op::Type::Add, 5);
}

} // namespace

int main() {
  testMatmul0();
  testRecompute0();
  testNoRecomputeWithAffine0();
  testNoFlow0();
  testComplexOp0();
  testComplexOp1();
  testComplexOp2();
  return 0;
}
