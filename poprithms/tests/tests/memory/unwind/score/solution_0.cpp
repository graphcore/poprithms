// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/unwind/error.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/path.hpp>
#include <poprithms/memory/unwind/solution.hpp>

namespace {

using namespace poprithms::memory::unwind;

void testSourceAtSink0() {

  //
  //   sink <== source
  //    |
  //  slice
  //    |
  //  slice
  //    |
  // flatten
  //    |
  //   out
  //
  //  We must confirm that `out` gets the correct Path.

  Graph g;
  const auto sink   = g.sink({6, 2});
  const auto source = g.source({6, 2});
  g.insertValuedPair(sink, source, 65.);

  const auto foo0 = g.slice(sink, {1, 0}, {5, 2});
  const auto foo1 = g.slice(foo0, {1, 0}, {3, 2});
  const auto out  = g.flatten(foo1);

  const Solution soln(g, {Path(source, Chain({6, 2}), sink)});

  Chain expected({6, 2});
  expected.slice({2, 0}, {4, 2});
  expected.flatten();
  expected.canonicalize();

  if (soln.inwardsPaths(out).size() != 1) {
    throw error("Expected 1 Path to output");
  }
  soln.inwardsPaths(out)[0].chain().confirmEqual(expected);
}

void testSourceMidSentence0() {

  //
  //
  //     sink
  //      |
  //  dimShuffle
  //      |
  //  dimShuffle
  //      |
  //      x1 <======== source.
  //      |
  //  dimShuffle
  //      |
  //  dimShuffle
  //
  // We check that all the intermediate Tensors have the correct layout, set
  // from source.
  //

  Graph g;
  const auto sink = g.sink({3, 4, 5, 6});
  const auto x0   = g.dimShuffle(sink, {{1, 2, 3, 0}});
  const auto x1   = g.dimShuffle(x0, {{1, 2, 3, 0}});
  const auto x2   = g.dimShuffle(x1, {{1, 2, 3, 0}});
  g.dimShuffle(x2, {{1, 2, 3, 0}});

  const auto source = g.source({5, 6, 3, 4});
  g.insertValuedPair(x1, source, 65.);

  Paths sPaths;

  Chain c({5, 6, 3, 4});
  c.dimShuffle({{2, 3, 0, 1}});
  sPaths.push_back(Path(source, c, sink));
  Solution soln(g, sPaths);
  // g.setPaths(soln);

  std::array<TensorId, 4> tIds{sink, x0, x1, x2};
  for (int64_t i = 0; i < 4; ++i) {
    TensorId tId = tIds[i];
    if (soln.inwardsPaths(tId).size() != 1) {
      throw error(
          "Chain of DimShuffles, expected each Tensor to have just 1 Path");
    }

    Chain expected({5, 6, 3, 4});
    expected.dimShuffle(Permutation({1, 2, 3, 0}).pow(i - 2));
    expected.canonicalize();
    if (soln.inwardsPaths(tId)[0].chain() != expected) {
      std::ostringstream oss;
      oss << "Error for TensorId " << tId << ", expected " << expected
          << " but observed " << soln.inwardsPaths(tId);
      throw error(oss.str());
    }
  }
}
} // namespace

int main() {
  testSourceAtSink0();
  testSourceMidSentence0();
  return 0;
}
