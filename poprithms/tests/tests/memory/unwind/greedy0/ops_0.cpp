// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/hosttensorhelper.hpp>
#include <poprithms/memory/unwind/solution.hpp>

namespace {

using namespace poprithms::memory::unwind;

void expandScoreTest0() {

  Graph g;

  auto x0 = g.sink({1, 4});
  auto x1 = g.expand(x0, {3, 4});
  auto x2 = g.barrier({}, {{3, 4}});
  g.insertValuedPair(x1, {x2, 0}, 10.);

  auto soln = Solution(g);

  auto toX1 = soln.inwardsPaths(x1);
  if (toX1.size() != 1) {
    throw poprithms::test::error("Expected just 1 path to sink");
  }
  Chain expected({3, 4});
  expected.slice({0, 0}, {1, 4});
  expected.expand({3, 4});
  toX1[0].chain().confirmEqual(expected.canonicalized());

  if (soln.getScore() != 0) {
    std::ostringstream oss;
    oss << "The use of the expand op was expected to result "
        << "in an underestimate of the score, as comparing the chain "
        << expected
        << " to the identity chain is false. The observed score was "
        << soln.getScore()
        << ", we expected to observe 0, and the true score is 10.*4 = 40. ";
    throw poprithms::test::error(oss.str());
  }
}

void multiUnwindTest0() {

  /**
   *
   *
   *          x2
   *           |
   *         barrier
   *           |
   * x0        xz       x1
   *  |       |  |       |
   *  +- cat -+  +- cat -+
   *      |          |
   *      x4         x5
   *      |          |
   *      +--- add --+
   *            |
   *          matmul (or something valuable).
   *
   * Both x0 and x1 have a path to a matmul input. To ensure they both get
   * their layouts set by the matmul, we must unwind through both input
   * indices of the add. Indeed, when unwinding through both indices is
   * enabled, x0 and x1 will both get layouts determined by the output of
   * add's target (matmul).
   *
   * Note that there is no guarantee that the output of add will actually be
   * the layout for the matmul, as we don't know how the backend actually
   * chooses an input to use for the add's output layout. This is the
   * slightly dodgey aspect about unwinding through multiple indices -- the
   * backend eventually must choose just one index but we don't specify which.
   * This means that the score returned by the Solution is possibly incorrect
   * when multiple indices are used for uwinding too.
   *
   * */

  auto test = [](std::vector<InIndex> uwInds) {
    Graph g;
    auto x0 = g.sink({1});
    auto x1 = g.sink({1});
    auto x2 = g.sink({1});
    auto x3 = g.source({1});
    g.insertValuedPair(x2, x3, 1.0);
    auto xz = TensorId(g.barrier({x2}, {{1}}), 0);
    auto x4 = g.concat({x0, xz}, 0);
    auto x5 = g.concat({xz, x1}, 0);
    auto x6 = g.sumLike({x4, x5}, uwInds, 10.);
    auto x7 = g.source({2});
    g.insertValuedPair(x6.out(), x7, 1000.);
    auto soln = Solution(g);

    if (soln.inwardsPaths(x0).size() != 1 ||
        soln.inwardsPaths(x1).size() != 1) {
      throw poprithms::test::error("Expected 1 path for each of x0 and x1");
    }

    if (std::find(uwInds.cbegin(), uwInds.cend(), 0) != uwInds.cend()) {
      if (soln.inwardsPaths(x0)[0].src() != x7) {
        throw poprithms::test::error("0 is unwindable, expect x7 source");
      }
    } else {
      if (soln.inwardsPaths(x0)[0].src() != xz) {
        throw poprithms::test::error("0 is not unwindable, expect xz source");
      }
    }

    if (std::find(uwInds.cbegin(), uwInds.cend(), 1) != uwInds.cend()) {
      if (soln.inwardsPaths(x1)[0].src() != x7) {
        throw poprithms::test::error("1 is unwindable, expect x7 source");
      }
    } else {
      if (soln.inwardsPaths(x1)[0].src() != xz) {
        throw poprithms::test::error("1 is not unwindable, expect xz source");
      }
    }
  };

  test({0});
  test({1});

  test({0, 1});
}

} // namespace

int main() {
  expandScoreTest0();
  multiUnwindTest0();
}
