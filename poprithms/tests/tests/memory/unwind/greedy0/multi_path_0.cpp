// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/memory/unwind/error.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/hosttensorhelper.hpp>
#include <poprithms/memory/unwind/solution.hpp>

namespace {

using namespace poprithms::memory::unwind;
using namespace poprithms::compute;

void test0() {

  Graph g;

  //
  //  [[ 6 7 0 1 ]
  //   [ 2 3 4 5 ]]
  const auto a = g.sink({2, 4});

  //  [[ . . 0 1 ]
  //   [ . . 4 5 ]]
  const auto b = g.identity(a);

  //  [[ 6 7 . . ]
  //   [ 2 3 . . ]]
  const auto c = g.reverse(a, Dimensions({0}));

  //  [[ . . 0 1 2 3 . . ]
  //   [ . . 4 5 6 7 . . ]]
  const auto d = g.concat({b, c}, 1);

  //  [[ 0 1 2 3 ]
  //   [ 4 5 6 7 ]]
  const auto e = g.slice({d}, {0, 2}, {2, 6});

  //  [[ 0 1 2 3 ]
  //   [ 4 5 6 7 ]]
  const auto f = g.source({2, 4});
  g.insertValuedPair(e, f, 10);

  std::cout << g << std::endl;

  const auto hosts = HostTensorHelper::arangeBarriers(g);
  const Solution s(std::move(g));

  HostTensorHelper::get(s, a, hosts)
      .assertAllEquivalent(
          host::Tensor::int64({2, 4}, {6, 7, 0, 1, 2, 3, 4, 5}));
}

void test1() {
  Graph g;

  // [[ 0 ]  or  [[ 1 ]
  //  [ 2 ]]      [ 3 ]]
  const auto a = g.sink({2, 1});

  // [[ 0 2 ]   or   [[ 1 3 ]
  //  [ 0 2 ]]        [ 1 3 ]]
  const auto b = g.concat({a, a}, 1);

  // [[ 0 1 ]
  //  [ 2 3 ]]
  const auto c = g.source({2, 2});
  g.insertValuedPair(b, c, 10.);

  const Solution s(std::move(g));

  if (s.getScore() != 2 * 10.) {
    std::ostringstream oss;
    oss << "Failed to compute correct score. "
        << "Expected half of the target layout to be matched. ";
    throw error(oss.str());
  }
}
} // namespace

int main() {

  test0();
  test1();
}
