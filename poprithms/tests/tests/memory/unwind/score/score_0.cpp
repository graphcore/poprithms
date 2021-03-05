// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <poprithms/memory/unwind/error.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/solution.hpp>

namespace {
using namespace poprithms::memory::unwind;

void test0() {

  for (auto withReverse : {true, false}) {

    //
    // sink0
    //  |
    // flatten
    //   |
    //  reverse (if withReverse)
    //    |
    //   x0     source
    //    .        |
    //    ..... source0
    //
    //
    // The unwind path always igores withReverse. So, if withReverse is true,
    // the unwind path does not result in a match between x0 and source so no
    // points are obtained.

    Graph g;

    const auto sink0 = g.sink0({10, 10});
    auto f0          = g.flatten(sink0);
    auto x0          = f0;
    if (withReverse) {
      x0 = g.reverse(f0, Dimensions({0}));
    }

    const auto source0 = g.source0({100});
    const auto value   = 2.;
    g.insertValuedPair(x0, source0, value);

    const auto p0 = g.getPath(source0, {Link(f0.opId(), 0, 0, false)}, sink0);

    Solution soln(g, {p0});

    const auto score    = soln.getScore();
    const auto expected = withReverse ? 0. : g.nelms(sink0) * value;
    if (score != expected) {
      std::ostringstream oss;
      oss << "Expected a score of " << expected << ", observed was " << score
          << ". This with withReverse=" << withReverse << '.';
      throw error(oss.str());
    }
  }
}

} // namespace

int main() {
  test0();
  return 0;
}
