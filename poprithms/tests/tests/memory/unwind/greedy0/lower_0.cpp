// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <memory>

#include <testutil/memory/nest/randomregion.hpp>
#include <testutil/memory/unwind/fullstate.hpp>
#include <testutil/memory/unwind/graph.hpp>
#include <testutil/memory/unwind/op.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/unwind/sumlike.hpp>

namespace {

using namespace poprithms;
using poprithms::common::multiout::InIndex;
using poprithms::memory::unwind::SumAttractions;

void basicTest0() {

  unwindtoy::Graph g;

  /**
   *                     +----dimShuffle --- (rhs) ---+
   *                     |                            |
   *  input (in5x4) -> slice -+                       v
   *                          |                       |
   *                          +---> sum  -> (lhs) -> matmul
   *                          |
   *  input (in5x1) ---->-----+
   *
   * */

  auto in5x4    = g.input({5, 4}, 1.0, "in5x4");
  auto slice5x2 = g.slice(in5x4, {0, 0}, {5, 2});
  auto in5x1    = g.input({5, 1}, 1.0, "in5x1");
  auto sum5x2   = g.sum({slice5x2, in5x1}, SumAttractions(10.0));
  g.matmul(sum5x2,
           g.dimShuffle(slice5x2, {{1, 0}}),
           unwindtoy::MatMulAttractions::Default().rhs(999.));

  // Priorities: matmul rhs > sum > linear mappers.

  unwindtoy::FullState fs(g);

  fs.lower();

  // Test that the layout of the slice of in5x4 used in the matmul
  // is set for the rhs of a matul:
  fs.createMappedSrc({"rhs_MatMul"}, 0)
      .dimShuffle({{1, 0}})
      .assertAllEquivalent(fs.mainLayout(in5x4).slice({0, 0}, {5, 2}));

  // Test that remainder of in5x4, the part which doesn't
  // go into the matmul, is mapped linearly:
  fs.createMappedSrc({"linear", "mapper", "in5x4"}, 0)
      .slice({0, 2}, {5, 4})
      .assertAllEquivalent(fs.mainLayout(in5x4).slice({0, 2}, {5, 4}));

  // Test that the layout of in5x1 is created with a layout
  // for a broadcast add:
  fs.createMappedSrc({"sumLike-reduce"}, 0)
      .assertAllEquivalent(fs.mainLayout(in5x1));
}

/**
 *
 *  in0 ---+
 *         |
 *         |
 *         +------> matmul ---> out
 *         |
 *         |
 *  in1 ---+
 *
 * By toggling the attraction values, we assert that we have
 *
 * 1) linear mappings of in0 and in1 where appropriate
 * 2) mappings set by the custom input creators (createLHSInput, etc).
 * 3) out having the same mapping as in0 and/or in1 when appropriate.
 *
 *
 * \param lin0: attraction between in0 and a linearly mapped tensor
 * \param lin1: attraction between in1 and a linearly mapped tensor
 * \param mma: the 4 attractions of a matmul.
 *
 * */

void testMatMulPreferences0(double lin0,
                            double lin1,
                            const unwindtoy::MatMulAttractions &mma) {

  unwindtoy::Graph g;

  auto in0 = g.input({5, 5}, lin0, "in0");
  auto in1 = g.input({5, 5}, lin1, "in1");
  auto out = g.matmul(in0, in1, mma);
  std::string mmName{"mmo100"};
  g.setName(out, mmName);

  unwindtoy::FullState fs(g);
  fs.lower();

  // left hand side layout:
  {

    // createLhsInput:
    if (mma.lhs() > mma.lhsOut() && mma.lhs() > lin0) {
      fs.createMappedSrc({"lhs_MatMul"}, 0)
          .assertAllEquivalent(fs.mainLayout(in0));
    }

    // match the output of the matmul to the lhs:
    else if (mma.lhsOut() > mma.lhs() && mma.lhsOut() > lin0) {
      fs.createMappedSrc({mmName}, 0).assertAllEquivalent(fs.mainLayout(in0));
    }

    // linearly map phs:
    else {
      fs.createMappedSrc({"linear", "mapper", "in0"}, 0)
          .assertAllEquivalent(fs.mainLayout(in0));
    }
  }

  // right hand side layout:
  {
    if (mma.rhs() > mma.rhsOut() && mma.rhs() > lin1) {
      fs.createMappedSrc({"rhs_MatMul"}, 0)
          .assertAllEquivalent(fs.mainLayout(in1));
    } else if (mma.rhsOut() > mma.rhs() && mma.rhsOut() > lin1) {
      fs.createMappedSrc({mmName}, 0).assertAllEquivalent(fs.mainLayout(in1));
    } else {
      fs.createMappedSrc({"linear", "mapper", "in1"}, 0)
          .assertAllEquivalent(fs.mainLayout(in1));
    }
  }
}

void testMultiUnwind0() {

  unwindtoy::Graph g;

  /**
   *
   *   in0 ----+-------+
   *           |       |
   *   (x0) slice  +-slice (the big slice, x1)
   *           |   |     |
   *           |   |     |
   *           |   |     |
   *           +--sum----+
   *               |
   *            +--+---------+
   *            |            |
   *            |           rhs
   *            |            |
   *         dimShuffle      |
   *            |            v
   *           lhs           |
   *            |            |
   *            +--- matmul -+
   *
   *
   * priorities:
   *  1) the dimShuffled input to matmul
   *  2) input 1 of sum -> reduced layout to input 0 of sum.
   *
   * */

  auto in0 = g.input({5, 4});
  auto x0  = g.slice(in0, {0, 0}, {5, 1});
  auto x1  = g.slice(in0, {0, 1}, {5, 4});

  // unwindable at indices 1 and 2 (dominating shapes)
  // Attraction is strong between inputs 0 and 1, so input 1
  // should set the layout of input 0.
  auto x2 =
      g.sum({x0, x1, x1}, SumAttractions({{1, 0, 50.}, {2, 0, 20.}}, 5));
  auto x3 = g.dimShuffle(x2, {{1, 0}});
  g.matmul(x3, x2, unwindtoy::MatMulAttractions::Default().lhs(1000));

  unwindtoy::FullState fs(g);
  fs.lower();

  // the big slice should have the same layout as
  // the (transpose) of the lhs input to the matmul:
  fs.mainLayout(in0)
      .slice({0, 1}, {5, 4})
      .assertAllEquivalent(
          fs.createMappedSrc({"lhs_MatMul"}, 0).dimShuffle({{1, 0}}));

  // the little slice should have a layout in preparation for
  // being added (to input 1) of the sum.
  fs.mainLayout(in0)
      .slice({0, 0}, {5, 1})
      .assertAllEquivalent(fs.createMappedSrc({"InIndex:1->0"}, 0));
}

void testMatMulPreferences0s() {
  // expect the specialised lhs and rhs creators to set layouts
  testMatMulPreferences0(1, 1, unwindtoy::MatMulAttractions::Default());

  // expect lhs input and output to have the same layout
  testMatMulPreferences0(
      1, 1, unwindtoy::MatMulAttractions::Default().lhsOut(1000));

  // expect lhs to mapped linearly.
  testMatMulPreferences0(1000, 1, unwindtoy::MatMulAttractions::Default());
  testMatMulPreferences0(
      1, 1, unwindtoy::MatMulAttractions::Default().rhsOut(1000));
  testMatMulPreferences0(1, 1000, unwindtoy::MatMulAttractions::Default());
}

void test3() {

  unwindtoy::Graph g;

  /**
   *           s0  s2  s1
   * x x x => x x x
   * x x x    x x x  x x x
   * x x x ========> x x x
   * x x x           x x x
   * x x x ====> x x x
   *
   *                     s2
   *  s1 ------+          |
   *           |          v
   *  s0 ->- matmul ---> sum
   *
   * What's interesting about this situation is that the output of matmul is
   * required before its inputs are available.
   *
   * Logic:
   * 1) to compute phi, in0 and in1 are required completely.
   * 2) the layout of in0 which s2 is a slice of is determined by the final
   *    sum, x1. But x1 requires the output of the matmul. So the matmul must
   *    be run a first time just to get layout of its output. There's nothing
   *    stopping us from doing this! At the poplar level, it will just be a
   *    poplar::program::Sequence which is not used, and poplar will prune it
   *    from the Graph.
   *
   * */

  auto in0 = g.input({5, 3});
  auto in1 = g.input({});
  auto phi = g.sum({in0, in1}, SumAttractions(4));
  auto s0  = g.slice(phi, {0, 0}, {2, 3});
  auto s1  = g.slice(phi, {1, 0}, {4, 3});
  auto s2  = g.slice(phi, {4, 0}, {5, 3});
  auto x0 =
      g.matmul(s0, s1, unwindtoy::MatMulAttractions::Default().lhs(10000));
  g.sum({x0, s2}, SumAttractions(4));
  unwindtoy::FullState fs(g);
  fs.lower();

  // The left-hand side input has highest preference for layout, so the full
  // slice s0 gets layed out to match the left-hand side. This includes the
  // bit which overlaps with s2.
  fs.mainLayout(in0)
      .slice({0, 0}, {2, 3})
      .assertAllEquivalent(fs.createMappedSrc({"lhs_MatMul"}, 0));

  fs.mainLayout(in0)
      .slice({2, 0}, {4, 3})
      .assertAllEquivalent(
          fs.createMappedSrc({"rhs_MatMul"}, 0).slice({1, 0}, {3, 3}));

  fs.mainLayout(in0)
      .slice({4, 0}, {5, 3})
      .assertAllEquivalent(fs.createMappedSrc(
          {"InIndex:0->1", "sumLike-reduce", "MatMul"}, 0));

  fs.mainLayout(in1).assertAllEquivalent(
      fs.createMappedSrc({"InIndex:0->1", "sumLike-reduce", "Input"}, 0));
}

void test4() {

  unwindtoy::Graph g;

  auto in0 = g.input({2, 2});
  auto in1 = g.input({2, 2});
  auto in2 = g.input({});

  auto x1 = g.matmul(in0, in1);
  auto x2 = g.matmul(x1, in0);
  auto x3 = g.matmul(x2, x1);
  g.sum({x3, in2, in2}, SumAttractions(4));

  unwindtoy::FullState fs(g);
  fs.lower();

  // Assert that the final path, the one to in2, appears after all of the
  // matmuls.
  bool pathAtEnd{false};
  bool atLeastOneMatMulFound{false};
  for (auto i : fs.scheduledSolution().schedule()) {
    if (fs.scheduledSolution().isPathToSink(i)) {
      pathAtEnd = true;
    } else {
      const auto opStr = g.op(fs.scheduledSolution().op(i)).str();
      if (opStr.find("MatMul") != std::string::npos) {
        atLeastOneMatMulFound = true;
        pathAtEnd             = false;
      }
    }
  }

  if (!atLeastOneMatMulFound) {
    throw poprithms::test::error(
        "Logic error, not a single MatMul found in schedule");
  }

  if (!pathAtEnd) {
    throw poprithms::test::error(
        "Expected Path to appear only after all of the MatMuls");
  }
}

void test5() {

  /**
   *
   * in0 -------------+
   *                  |
   *                  +--- matmul
   *                  |
   * in1 -- expand ---+
   *
   * We test that the layout of in1 is the lower slice of
   * the optimal rhs input of a matmul. This choice is implemented in
   * poprithms::memory::unwind::Expand::bwd.
   *
   * Note that the reported score is not accurate in this case.
   * */

  unwindtoy::Graph g;
  auto in0 = g.input({2, 3});
  auto in1 = g.input({1, 4});
  auto x   = g.expand(in1, {3, 4});
  auto out = g.matmul(in0, x);
  unwindtoy::FullState fs(g);

  fs.lower();

  fs.mainLayout(in1).assertAllEquivalent(
      fs.createMappedSrc({"rhs", "MatMul"}, 0).slice({0, 0}, {1, 4}));

  {
    const auto mma = g.matMulAttractions(out.opId());

    // This is the correct score: 1 slice of the rhs matches
    // the matmul input target, and the entire lhs inputs matches.
    const auto expected0 = mma.rhs() * 1 * 4 + mma.lhs() * 2 * 3;
    (void)expected0;

    // With the current score calculator, this is the score we get.
    // The reason no points are obtained from rhs is that the chain
    // settSample -> expand does not match the empty chain.
    const auto expected1 = mma.lhs() * 2 * 3;

    const auto observed = fs.scheduledSolution().getScore();
    if (expected1 != observed) {
      std::ostringstream oss;
      oss << "Expected score of " << expected1 << ", but observed "
          << observed << '.';
      throw poprithms::test::error(oss.str());
    }
  }
}

} // namespace

int main() {
  basicTest0();
  testMatMulPreferences0s();
  testMultiUnwind0();
  test3();
  test4();
  test5();
  return 0;
}
