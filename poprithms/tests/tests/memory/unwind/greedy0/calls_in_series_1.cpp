// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <map>

#include <poprithms/memory/unwind/error.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/solution.hpp>

int main() {

  using namespace poprithms::memory::unwind;

  //
  // outer graph is:
  //
  //     sinkOut
  //       |
  //     reshape
  //       |
  //     call
  //       |
  //     [out0]
  //       |
  //     call
  //       |
  //     [out1] <========= source
  //
  //  and inner graph is:
  //
  //     sinkInn
  //        |
  //     reshape
  //        |
  //    dimShuffle
  //
  //
  // In total there are 4 sinks: the 2 starting Tensors in the Graphs (one in
  // the outer, and one in the inner scope), and the outputs of the calls.
  //
  // This is testing the ability to unwind backwards through 2 entire call
  // ops.
  //

  Graph g;
  const auto sinkInn = g.sink({4, 5});
  const auto a0      = g.reshape(sinkInn, {5, 4});
  const auto b0      = g.dimShuffle(a0, {{1, 0}});

  const auto sinkOut = g.sink({20});
  const auto a1      = g.reshape(sinkOut, {4, 5});

  double call0val = 1.0;
  double call1val = 10.0;
  if (call0val >= call1val) {
    throw error("This test requires call1val to be larger");
  }
  const auto out0 = g.call({a1}, {sinkInn}, {b0}, call0val)[0];
  const auto out1 = g.call({out0}, {sinkInn}, {b0}, call1val)[0];

  const auto sourceId = g.source({4, 5});
  g.insertValuedPair(sourceId, out1, 5.);

  auto s = Solution(g);

  std::map<TensorId, Chain> expected;
  expected.emplace(out1, Chain({4, 5}));
  expected.emplace(b0, Chain({4, 5}));
  Chain ea0({4, 5});
  ea0.dimShuffle({{1, 0}});
  expected.emplace(a0, ea0);

  auto eSinkInn = ea0;
  eSinkInn.reshape({4, 5});

  expected.emplace(sinkInn, eSinkInn);

  // out0 is copied into the second call, so it gets its value set from a0.
  expected.emplace(out0, eSinkInn);

  for (auto tId : g.tensorIds()) {
    const auto found = expected.find(tId);
    if (found != expected.cend()) {
      const Path expe(sourceId, found->second, tId);
      if (s.inwardsPaths(tId) != Paths({expe})) {
        std::ostringstream oss;
        oss << "Path for Tensor " << tId << " is not as expected. "
            << "Expected \n"
            << expe << ", observed \n"
            << s.inwardsPaths(tId);
        throw error(oss.str());
      }
    }
  }

  return 0;
}
