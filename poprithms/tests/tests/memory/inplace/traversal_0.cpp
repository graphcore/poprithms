// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <array>
#include <iostream>
#include <sstream>

#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/common/multiout/traversal.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

namespace {

using namespace poprithms::memory::inplace;

void testTraversal0() {
  Graph graph;
  const auto v0 = Tensor::variable(graph, {3});
  graph.multi({v0.id(), v0.id(), v0.id()}, {{}, {}, {}, {}}, {});
  if (poprithms::common::multiout::depthFirstForward(
          graph, {v0.id()}, [](auto) { return true; })
          .size() != 12) {
    throw poprithms::test::error("3 inputs, 4 outputs: 12 paths.");
  }
}

// A wrapper around a Graph, which provides a custom 'neighbors' method. For
// forward traversal, this just returns the consumers of an ops output
// tensors.
class ForwardTraverse {
public:
  ForwardTraverse(const Graph &g) : g_(g) {}
  const Graph &g_;
  OpIds neighbors(OpId opId) const {
    OpIds opIds;
    for (auto t : g_.outTensorIds(opId)) {
      for (auto c : g_.consumptionIds(t)) {
        opIds.push_back(c.opId());
      }
    }
    return opIds;
  }
};

void visitAndAssert(const Graph &graph,
                    const OpIds &starts,
                    const OpIds &terminals,
                    const OpIds &expected) {

  ForwardTraverse fwdTraverse(graph);

  auto terminate = [terminals](OpId opId) {
    return std::find(terminals.cbegin(), terminals.cend(), opId) !=
           terminals.cend();
  };

  // starting at x0 and x4, terminate (and do not include) at x2.
  auto visited =
      poprithms::common::multiout::depthFirst<ForwardTraverse, OpId>(
          std::move(fwdTraverse), starts, terminate);

  std::sort(visited.begin(), visited.end());
  if (visited != expected) {
    using poprithms::common::multiout::operator<<;
    std::ostringstream oss;
    oss << "Expected the depth first tensors to be " << expected << ", not "
        << visited << '.';
  }
}

void testDepthFirst0() {

  {
    // x0 -> x1 -> x2
    // x3 -> x4 -> x5.

    Graph graph;
    const auto x0 = Tensor::variable(graph, {});
    const auto x1 = x0.reshape({1});
    const auto x2 = x1.reshape({1, 1});
    const auto x3 = Tensor::variable(graph, {1, 1});
    const auto x4 = x3.reshape({1});
    const auto x5 = x4.reshape({});
    visitAndAssert(graph,
                   {x0.opId(), x1.opId()},
                   {x2.opId()},
                   {x0.opId(), x1.opId(), x4.opId(), x5.opId()});
  }

  {

    // x0 -> x1 -> c2 -> c3
    //    ->    ->    -> c4.
    //          ->
    //          ->

    Graph graph;
    const auto x0  = Tensor::variable(graph, {});
    const auto x1s = x0.multi(graph, {x0, x0}, {{1}, {2}, {3}, {3}}, {});
    const auto c2  = x0.concat(x1s, 0);
    const auto c3  = c2.flatten();
    const auto c4  = c3.flatten();

    OpIds expected{x0.opId(), x1s[0].opId(), c2.opId(), c3.opId()};
    visitAndAssert(graph, {x0.opId()}, {c4.opId()}, expected);
  }
}

void testTensorTraversalBase(const TensorIds &starts,
                             TensorIds observed,
                             TensorIds expected) {
  std::sort(observed.begin(), observed.end());
  std::sort(expected.begin(), expected.end());
  if (observed != expected) {
    std::ostringstream oss;
    oss << "Failed to detect the correct set of TensorIds in traversal from "
        << starts << '.' << " The returned set is \n   " << observed
        << ", not \n   " << expected << '.';
    throw poprithms::test::error(oss.str());
  }
}

void testTensorTraversal0() {
  Graph graph;
  auto x0 = Tensor::variable(graph, {1, 1, 1});
  auto x1 = Tensor::variable(graph, {1, 1, 1});
  auto c0 = Tensor::concat({x0, x1}, 0);
  auto c1 = Tensor::concat({x0, x1}, 0);
  auto r0 = c0.reshape({2});
  auto r1 = c1.reshape({2});
  auto o0 = Tensor::aliasGate({r0, r1});

  {
    TensorIds starts{{x0.id()}};
    auto expected = Tensor::tensorIds({x0, c0, c1, r0, r1, o0});
    auto observed = poprithms::common::multiout::depthFirstForwardTensors(
        graph, starts, [](const TensorId &) { return true; });
    testTensorTraversalBase(starts, observed, expected);
  }

  {
    TensorIds starts{x0.id(), x1.id()};
    auto observed = poprithms::common::multiout::depthFirstForwardTensors(
        graph, starts, [&graph](const TensorId &tId) {
          return graph.shape(tId) != Shape({2});
        });
    auto expected = Tensor::tensorIds({x0, x1, c0, c1});
    testTensorTraversalBase(starts, observed, expected);
  }

  {
    TensorIds starts{r0.id()};
    TensorIds expected{r0.id(), c0.id()};
    auto observed = poprithms::common::multiout::depthFirstBackwardTensors(
        graph, starts, [&graph](const TensorId &tId) {
          return graph.nInTensors(tId.opId()) != 0;
        });
    testTensorTraversalBase(starts, observed, expected);
  }

  {
    TensorIds starts{r0.id()};
    TensorIds expected{r0.id(), c0.id(), x0.id(), x1.id()};
    auto observed = poprithms::common::multiout::depthFirstBackwardTensors(
        graph, starts, [](const TensorId &) { return true; });
    testTensorTraversalBase(starts, observed, expected);
  }
}

} // namespace

int main() {
  testTraversal0();
  testDepthFirst0();
  testTensorTraversal0();
  return 0;
}
