// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/solution.hpp>

// Based on the example and discussion in
// https://phabricator.sourcevertex.net/T30324

namespace {
// The score for getting the weights input to a matmul to have the layout
// determined by poplibs API. Per element.
constexpr double valPoplibsMatmulRHS{10.};

// The score for getting the activations input to a matmul to have the
// layout determined by poplibs API. Per element.
constexpr double valPoplibsMatmulLHS{9.};

// The score per element obtained if the Tensor copied out of a call has the
// same layout as the Tensor in the main graph to which it is copied.
constexpr double valCopyOutSame{2.};

// I'm not sure where the input to main Graph gets its layout from in the
// exampl in T30324
const double valLinearMapMainInput{1.};

// clang-format off
//
// When valCopyInSame = 2.0 and valMatmulOutAndInSame = 8.5, printing the Graph gives:
//
//
//    OpId  Name                   OpType   InTensors  Shape  Attractors
//    ----- ---------------------- -------- ---------- ------ -------------------------------------------------------------------
//    0     matmul activation in   Sink     ()         (1)    (((op=1,v=9),(op=4,v=8.5),(op=7,v=2.5),(op=8,v=2.5),(op=9,v=2.5)))
//    1     poplibs create LHS     Source   ()         (1)    (((op=0,v=9)))
//    2     matmul weights         Sink     ()         (1)    (((op=3,v=10)))
//    3     poplibs create RHS     Source   ()         (1)    (((op=2,v=10)))
//    4     matmul activation out  Source   ()         (1)    (((op=0,v=8.5),(op=8,v=2),(op=9,v=2),(op=10,v=2)))
//    5     input to main          Sink     ()         (1)    (((op=6,v=1)))
//    6     input target (linear)  Source   ()         (1)    (((op=5,v=1)))
//    7     a                      Barrier  ((op=5))   (1)    (((op=0,v=2.5)))
//    8     b                      Sink     ()         (1)    (((op=4,v=2),(op=0,v=2.5)))
//    9     c                      Sink     ()         (1)    (((op=4,v=2),(op=0,v=2.5)))
//    10    d                      Sink     ()         (1)    (((op=4,v=2)))
//
// clang-format on
//

/**
 *
 * \param valCopyInSame
 *        The score per element obtained if the Tensor copied into a call has
 *        the same layout as the Tensor in the subgraph to which it is copied.
 *
 * \param valMatmulOutAndInSame
 *        The score for getting the activations input to a matmul to have the
 *        same layout as the matmul's output. This (like all scores) is a per
 *        element score. Note that this score is concerned only with the
 *        actual matmul, the global benefits of potentially fewer copies
 *        elsewhere are not managed by this score.
 *
 * */

void test(double valCopyInSame, double valMatmulOutAndInSame) {

  using namespace poprithms::memory::unwind;

  Graph g;

  // Subgraph input:
  const auto actInn = g.sink({1}, "matmul activation in");

  // Subgraph input target layout (createMatMulLHS target)
  const auto actInnSource = g.source({1});

  // Weights of matmul
  const auto weightInn       = g.sink({1}, "matmul weights");
  const auto weightInnSource = g.source({1});
  g.insertValuedPair(weightInnSource, weightInn, valPoplibsMatmulRHS);

  // matmul output. Its layout is assumed to be independent of ins (T32143).
  const auto mmOut = g.source({1}, "matmul activation out");

  // How valuable is it for actIn to have the layout of actInSource?
  g.insertValuedPair(actInn, actInnSource, valPoplibsMatmulLHS);

  // How good is it if the matmul input and output have same layout?
  g.insertValuedPair(mmOut, actInn, valMatmulOutAndInSame);

  const auto in0       = g.sink({1}, "input to main");
  const auto in0source = g.source({1}, "input target (linear)");
  g.insertValuedPair(in0, in0source, valLinearMapMainInput);

  // embedding output.
  TensorId a{g.barrier({in0}, {{1}}), 0};

  // calls.
  const auto b =
      g.call({a}, {actInn}, {mmOut}, {valCopyInSame}, {valCopyOutSame})[0];
  const auto c =
      g.call({b}, {actInn}, {mmOut}, {valCopyInSame}, {valCopyOutSame})[0];
  const auto d =
      g.call({c}, {actInn}, {mmOut}, {valCopyInSame}, {valCopyOutSame})[0];

  g.setName(a.opId(), "a");
  g.setName(b.opId(), "b");
  g.setName(c.opId(), "c");
  g.setName(d.opId(), "d");

  Solution soln(std::move(g));

  // std::cout << g << std::endl;

  // 1) layout of activation into matmul
  if (valPoplibsMatmulLHS > valMatmulOutAndInSame) {
    if (soln.inwardsPaths(actInn)[0].src() != actInnSource) {
      std::ostringstream oss;
      oss << "The value of using the poplibs API to set the activation input "
          << "to a matmul is set to valPoplibsMatmulLHS="
          << valPoplibsMatmulLHS
          << ". The value for having the layouts of the matmul input and "
             "outputs be the same is set to valMatmulOutAndInSame="
          << valMatmulOutAndInSame
          << ". Using the Greedy0 algorithm, we therefore expect "
          << "the poplibs API to be used, that is we "
          << "expect the activation input to have the same layout as "
          << "actInnSource. ";
      throw poprithms::test::error(oss.str());
    }
  } else {
    if (soln.inwardsPaths(actInn)[0].src() != mmOut) {
      std::ostringstream oss;
      oss << "The value of using the poplibs API to set the activation input "
          << "to a matmul is set to valPoplibsMatmulLHS="
          << valPoplibsMatmulLHS
          << ". The value for having the layouts of the matmul input and "
             "outputs be the same is set to valMatmulOutAndInSame="
          << valMatmulOutAndInSame
          << ". Using the Greedy0 algorithm, we therefore expect "
          << " the input and outputs to have ths layout. ";
      throw poprithms::test::error(oss.str());
    }
  }

  // 2) b and c, the intermediate activations in the main graph.
  if (soln.inwardsPaths(b)[0].src() != soln.inwardsPaths(c)[0].src()) {
    std::ostringstream oss;
    oss << "The choice of layout for " << b << " and " << c
        << " (b and c) should always match. ";
    throw poprithms::test::error(oss.str());
  }
  if (valCopyInSame > valCopyOutSame) {
    if (soln.inwardsPaths(b)[0].src() != soln.inwardsPaths(actInn)[0].src()) {
      std::ostringstream oss;
      oss << "We have that copy into calls are more important "
          << "than copies out of calls. "
          << "Therefore expect b and c to have the same layout as "
          << "the matmul input. ";
      throw poprithms::test::error(oss.str());
    }
  } else {
    if (soln.inwardsPaths(b)[0].src() != soln.inwardsPaths(mmOut)[0].src()) {
      std::ostringstream oss;
      oss << "We have that copy into calls are less important "
          << "than copies out of calls. "
          << "Therefore expect b and c to have the same layout as "
          << "the matmul output. ";
      throw poprithms::test::error(oss.str());
    }
  }
}
} // namespace

int main() {
  // 1) copy in more valuable than copy in, and
  // 2) having same layout for matmul input and output is MORE important than
  //    using poplibs matmul input creator.
  test(valCopyOutSame + 0.5, valPoplibsMatmulLHS + 0.5);

  // 1) copy out more valuable than copy in, and
  // 2) having same layout for matmul input and output is less important than
  //    using poplibs matmul input creator.
  test(valCopyOutSame - 0.5, valPoplibsMatmulLHS - 0.5);

  // 1) copy out more valuable than copy in, and
  // 2) having same layout for matmul input and output is MORE important than
  //    using poplibs matmul input creator.
  test(valCopyOutSame - 0.5, valPoplibsMatmulLHS + 0.5);

  // 1) copy out less valuable than copy in, and
  // 2) having same layout for matmul input and output is LESS important than
  //    using poplibs matmul input creator.
  test(valCopyOutSame + 0.5, valPoplibsMatmulLHS - 0.5);

  return 0;
}
