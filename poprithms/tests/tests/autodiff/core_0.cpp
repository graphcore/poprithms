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

  void assertNone(Op::Type t) { assertCount(t, 0); }
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

  // Id Type       Ins                    nOut  name
  // -- ----       ---                    ----  ----
  // 0  Variable   ()                     1     v0
  // 1  Variable   ()                     1     v1
  // 2  Matmul     ((op=0),(op=1))        1     mm0
  // 3  Variable   ()                     1     checkpoint/(op=0)
  // 4  Variable   ()                     1     checkpoint/(op=1)
  // 5  Variable   ()                     1     grad-in-of/(op=2)
  // 6  MatmulGrad ((op=5),(op=3),(op=4)) 2     grad-of-op-2-inputs-(0,1)

  Getter get0(testGraph);

  // 2 checkpoints:
  auto cp0 =
      get0({}, true, Op::Type::Variable, Autodiff::genCheckpointName({0, 0}));

  auto cp1 =
      get0({}, true, Op::Type::Variable, Autodiff::genCheckpointName({1, 0}));

  // no zeros,
  get0.assertNone(Autodiff::genInitGradName({0, 0}));
  get0.assertNone(Autodiff::genInitGradName({1, 0}));
  get0.assertNone(Op::Type::Zero);

  // 1 variable, "grad in of"
  auto gIn =
      get0({}, true, Op::Type::Variable, Autodiff::genInGradName({2, 0}));

  // 1 matmul grad, with 3 inputs.
  auto gmm =
      get0({{gIn, 0}, {cp0, 0}, {cp1, 0}}, false, Op::Type::MatmulGrad, "");
  (void)gmm;
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
  //  Type        Ins             outsRequired flows  name
  //  ----        ---             ------------ -----  ----
  //  Variable    ()              ()           ()     v0
  //  Unknown     ((op=0))        (0)          (0<-0) x0
  //  Unknown     ((op=1))        (0)          (0<-0) x1
  //  Variable    ()              ()           ()     checkpoint/(op=0)
  //  Unknown     ((op=3))        (0)          (0<-0) rerun/1
  //  Unknown     ((op=4))        (0)          (0<-0) rerun/2
  //  Variable    ()              ()           ()     grad-in-of/(op=2)
  //  UnknownGrad ((op=6),(op=5)) ()           ()     grad-of-op-2-inputs-(0)
  //  UnknownGrad ((op=7),(op=4)) ()           ()     grad-of-op-1-inputs-(0)

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
  x.assertNone(Op::Type::Zero);

  // grad in:
  TensorId gIn(x({}, true, Op::Type::Variable, Autodiff::genInGradName(x1)),
               0);

  // gradient of x0:
  TensorId ingrad1(x({gIn, recomp1}, false, Op::Type::UnknownGrad, ""), 0);

  // gradient of the traget, v0:
  TensorId ingrad0(x({ingrad1, recomp0}, false, Op::Type::UnknownGrad, ""),
                   0);
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

  std::cout << testGraph << std::endl;
  getter.assertNone(Autodiff::genRerunName(v0.opId()));
  getter.assertNone(Autodiff::genRerunName(x0.opId()));
  getter.assertNone(Autodiff::genCheckpointName(x0));
  getter.assertNone(Autodiff::genRerunName(x1.opId()));
}

void testNoFlow0() {
  TestGraphInfo testGraph;
  auto v0 = testGraph.insertNoFlow({}, "v0", Op::Type::Variable);

  // something like out(x) = (random(), largestFactor(int(x)))
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
  //  Id Type     Ins      nOut insRequired outsRequired name
  //  -- ----     ---      ---- ----------- ------------ ----
  //  0  Variable ()       1    ()          ()           v0
  //  1  Unknown  ((op=0)) 2    (0)         (0,1)        x0
  //  2  Variable ()       1    ()          ()           checkpoint/(op=0)
  //  3  Variable ()       1    ()          ()           checkpoint/(op=1)
  //  4  Variable ()       1    ()          ()           checkpoint/(op=1,out=1)
  //  5  Variable ()       1    ()          ()           grad-in-of/(op=1)
  //  6  Zero     ()       1    ()          ()           
  //
  // clang-format on
  //
  getter.assertCount(Op::Type::Add, 0);
  getter.assertCount(Op::Type::UnknownGrad, 0);

  // The gradient of v0.
  getter.assertCount(Op::Type::Zero, 1);
}

void testComplexOp0() {

  std::cout << "\ntestComplexOp0\n" << std::endl;

  //      +---- flow ---------> .... < gradient in
  //      |
  // x0 --+---- flow   ------->  ... < gradient in
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
  //
  //  Id Type        Ins              nOut outsReq flows       name
  //  -- ----        ---              ---- ------- -----       ----
  //  0  Variable    ()               1    ()      ()          v0
  //  1  Unknown     ((op=0))         3    (0,1,2) (0<-1,0<-0) x1
  //  2  Unknown     ((op=1))         1    (0)     (0<-0)      x10
  //  3  Unknown     ((op=1,out=1))   1    (0)     (0<-0)      x11
  //  4  Unknown     ((op=1,out=2))   1    (0)     (0<-0)      x12
  //  5  Variable    ()               1    ()      ()          checkpoint/(op=0)
  //  6  Unknown     ((op=5))         3    (0,1,2) (0<-1,0<-0) rerun/1
  //  7  Unknown     ((op=6))         1    (0)     (0<-0)      rerun/2
  //  8  Variable    ()               1    ()      ()          grad-in-of/(op=2)
  //  9  UnknownGrad ((op=8),(op=7))  1    ()      ()          grad-of-op-2-inputs-(0)
  //  10 Zero        ()               1    ()      ()          init-grad-of(op=1,out=1)
  //  11 UnknownGrad ((op=9),(op=10), 1    ()      ()          grad-of-op-1-inputs-(0)
  //                  (op=6),(op=6,out=1),
  //                  (op=6,out=2))
  //
  // clang-format on

  TestGraphMutator a(testGraph);
  Autodiff(g, testGraph, a);
  Getter getter(testGraph);

  std::cout << testGraph << std::endl;

  // we expect exactly 1 initialization (zero) op, for the gradient of the
  // output 1 of op 1.
  getter.assertCount(Op::Type::Zero, 1);
  getter({}, true, Op::Type::Zero, Autodiff::genInitGradName({x1, 1}));

  // Checks for recompute:
  TensorId cp{
      getter({}, true, Op::Type::Variable, Autodiff::genCheckpointName(x0)),
      0};
  TensorId recomp0{
      getter({cp}, true, Op::Type::Unknown, Autodiff::genRerunName(x1)), 0};
  getter({recomp0}, true, Op::Type::Unknown, Autodiff::genRerunName(x10));

  // 2 gradients, 1 for x0 and 1 for the 0'th output of x1.
  getter.assertCount(Op::Type::UnknownGrad, 2);
}

void testComplexOp1() {

  // How gradients flow in thie example (lines within dotted squares).
  //
  //                       "multi" op
  //                  . . . . . . . . .
  //           +--> 0 .  <---+----    . -------+
  //           |      .      |        .        |
  //           +--> 1 .      +----    . ---+   |
  //           |      . . . . . . . . .    |   |
  // input  ---+                           v   v
  //           |                           1   2
  //           |       . . . . . . . . . . .  . . .
  //           +---> 0 . <---+             ^   ^  .
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
                                         // no gradients flow to input 1.
                                         {{OutIndex{0}, 0}, {1, 0}},
                                         "multi"));

  const auto loss =
      testGraph.insert(Op({input, {multi, 0}, {multi, 1}, input},
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
  //
  // clang-format off
//
// Id Type        Ins                                             nOut insRequired outsRequired flows            name
// -- ----        ---                                             ---- ----------- ------------ -----            ----
// 0  Variable    ()                                              1    ()          ()           ()               input
// 1  Unknown     ((op=0),(op=0))                                 2    ()          (0,1)        (0<-0,0<-1)      multi
// 2  Unknown     ((op=0),(op=1),(op=1,out=1),(op=0))             1    ()          (0)          (0<-0,1<-0,2<-0) loss
// 3  Variable    ()                                              1    ()          ()           ()               checkpoint/(op=0)
// 4  Variable    ()                                              1    ()          ()           ()               checkpoint/(op=2)
// 5  Unknown     ((op=3),(op=3))                                 2    ()          (0,1)        (0<-0,0<-1)      rerun/1
// 6  Variable    ()                                              1    ()          ()           ()               grad-in-of/(op=2)
// 7  UnknownGrad ((op=6),(op=4))                                 3    ()          ()           ()               grad-of-op-2-inputs-(0,1,2)
// 8  UnknownGrad ((op=7,out=1),(op=7,out=2),(op=5),(op=5,out=1)) 1    ()          ()           ()               grad-of-op-1-inputs-(0)
// 9  Add         ((op=7),(op=8))                                 1    ()          ()           (0<-1,0<-0)      Add              
//
  // clang-format on
  //

  std::cout << testGraph << std::endl;

  // no zero gradients required:
  getter.assertCount(Op::Type::Zero, 0);

  // One Add at the end to create the gradient of the input from the 2 paths.
  getter.assertCount(Op::Type::Add, 1);

  // Complete rerun:

  auto cp0 = getter(
      {}, true, Op::Type::Variable, Autodiff::genCheckpointName(input));
  auto cp1 = getter(
      {}, true, Op::Type::Variable, Autodiff::genCheckpointName({loss, 0}));
  // must rerun multi, as its outputs are needed to compute gradients.
  auto r0 = getter({{cp0, 0}, {cp0, 0}},
                   true,
                   Op::Type::Unknown,
                   Autodiff::genRerunName(multi));
  // the promised gradient in:
  auto gIn = getter(
      {}, true, Op::Type::Variable, Autodiff::genInGradName({loss, 0}));
  // run the loss grad using the output and the input gradient. {0,1,2}
  // because these are the indices which the loss propagates gradient to.
  auto lGrad = getter({{gIn, 0}, {cp1, 0}},
                      false,
                      Op::Type::UnknownGrad,
                      Autodiff::genGradInsName(loss, {0, 1, 2}));
  // run the multi op grad.  use both of the outputs and both of the output
  // grads. The outputs grads were created by loss grad (no summing required,
  // as they're singleton sums).
  auto mGrad = getter({{r0, 0}, {r0, 1}, {lGrad, 1}, {lGrad, 2}},
                      false,
                      Op::Type::UnknownGrad,
                      Autodiff::genGradInsName(multi, {0}));
  // and finally, the sum to get the gradient of input.
  getter({{lGrad, 0}, {mGrad, 0}}, false, Op::Type::Add, "Add");
}

// Like complexOp1, but
// 1) flows through complex are modified.
// 2) input order to multi changed.
void testComplexOp2() {

  // How gradients flow in thie example (lines within dotted squares).
  //
  //                       "multi" op
  //                  . . . . . . . . .
  //           +----0 .  <---+----    . -------+
  //           |      .      |        .        |
  //           +----1 .  <---+        . ---+   |
  //           |      . . . . . . . . .    |   |
  // input  ---+                           v   v
  //           |                           0   1
  //           |       . . . . . . . . . . .  . . .
  //           +---> 2 .                   ^   ^  .
  //           |       .                   |   |  .
  //           +---> 3 . <---+-------------+---+  . ---> loss tensor
  //                   .                          .
  //                   . . . . . . . . . . .  . . .
  //                          "loss" op
  //

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
                          {{OutIndex(0), 0}, {0, 1}, {0, 3}},
                          "loss"));

  const auto g =
      guide::Objective::outOfGraph({{loss, 0}}, {input, {loss, 0}}, {input});
  TestGraphMutator a(testGraph);
  Autodiff(g, testGraph, a);
  Getter getter(testGraph);

  //
  // clang-format off
//
// 0  Variable    ()                                  1    ()          ()           ()               input
// 1  Unknown     ((op=0),(op=0))                     2    ()          (0,1)        (0<-0,1<-0)      multi
// 2  Unknown     ((op=1),(op=1,out=1),(op=0),(op=0)) 1    ()          (0)          (0<-0,1<-0,2<-0) loss
// 3  Variable    ()                                  1    ()          ()           ()               checkpoint/(op=0)
// 4  Variable    ()                                  1    ()          ()           ()               checkpoint/(op=2)
// 5  Unknown     ((op=3),(op=3))                     2    ()          (0,1)        (0<-0,1<-0)      rerun/1
// 6  Variable    ()                                  1    ()          ()           ()               grad-in-of/(op=2)
// 7  UnknownGrad ((op=6),(op=4))                     3    ()          ()           ()               grad-of-op-2-inputs-(0,1,3)
// 8  UnknownGrad ((op=7),(op=5),(op=5,out=1))        2    ()          ()           ()               grad-of-op-1-inputs-(0,1)
// 9  Add         ((op=7,out=2),(op=8))               1    ()          ()           (0<-1,0<-0)      Add
// 10 Add         ((op=9),(op=8,out=1))               1    ()          ()           (0<-1,0<-0)      Add
//
  // clang-format on
  //

  std::cout << testGraph << std::endl;

  getter.assertCount(Op::Type::Zero, 0);

  // expect 2 adds: as there are 3 paths from input to loss.
  getter.assertCount(Op::Type::Add, 2);

  // complete rerun. The first 4 checks are exactly as before (in ComplexOp1)
  auto cp0 = getter(
      {}, true, Op::Type::Variable, Autodiff::genCheckpointName(input));
  auto cp1 = getter(
      {}, true, Op::Type::Variable, Autodiff::genCheckpointName({loss, 0}));
  auto r0  = getter({{cp0, 0}, {cp0, 0}},
                   true,
                   Op::Type::Unknown,
                   Autodiff::genRerunName(multi));
  auto gIn = getter(
      {}, true, Op::Type::Variable, Autodiff::genInGradName({loss, 0}));

  // lGrad and mGrad are different:
  auto lGrad = getter({{gIn, 0}, {cp1, 0}},
                      false,
                      Op::Type::UnknownGrad,
                      Autodiff::genGradInsName(loss, {0, 1, 3}));
  getter({{r0, 0}, {lGrad, 0}, {r0, 1}},
         false,
         Op::Type::UnknownGrad,
         Autodiff::genGradInsName(multi, {1, 0}));
}

} // namespace

int main() {
  std::cout << "testMatMul0" << std::endl;
  testMatmul0();
  std::cout << "testRecompute0" << std::endl;
  testRecompute0();
  std::cout << "testNoRecomputeWithAffine0" << std::endl;
  testNoRecomputeWithAffine0();
  std::cout << "testNoFlow0" << std::endl;
  testNoFlow0();
  std::cout << "testComplexOp0" << std::endl;
  testComplexOp0();
  std::cout << "testComplexOp1" << std::endl;
  testComplexOp1();
  std::cout << "testComplexOp2" << std::endl;
  testComplexOp2();
  return 0;
}
