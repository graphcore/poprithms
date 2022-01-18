// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <sstream>

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/hosttensorhelper.hpp>
#include <poprithms/memory/unwind/solution.hpp>

namespace {

using namespace poprithms::memory::unwind;
using namespace poprithms::compute;

void testOrder0() {

  Graph g;

  Shape s0({5, 7});

  // create a sink (and a source to create a defalut/'linear' layout for this
  // sink). The strength of attraction of default source is #vLin.
  auto getSink = [&g, &s0](const std::string &name, double vLin = 0.1) {
    auto source = g.source(s0, name + "_source");
    auto sink   = g.sink(s0, name + "_sink");
    g.insertValuedPair(source, sink, vLin);
    return sink;
  };

  // this models something like (a + b).reduce()
  // The add is unwindable through index 0.
  auto getNext = [&g, &s0](TensorId a, TensorId b, const std::string &name) {
    auto sum_ = g.sumLike({a, b}, 0, 100.).out();
    // modelling the reduce, although the shape of the output is the shape of
    // the input.
    TensorId id{g.barrier({sum_}, {s0}, name + "_reduction_"), 0};
    return id;
  };

  // We are modelling:
  //
  // sink04 = (((sink00 + sink01).reduce() + sink02).reduce() +
  //           + sink03).reduce()
  // sink14 = (((sink10 + sink11).reduce() + sink12).reduce() +
  //           + sink13).reduce()
  // finale = sink04 + sink14.
  //
  //
  //  sink00
  //   |      sink01
  //   |       |
  //  add------+
  //   |
  //  reduce  sink02
  //   |       |
  //  add------+
  //   |
  //  reduce  sink03
  //   |       |
  //  add------+           ;
  //   |                   .
  //  reduce               .
  //   |                   |
  //   +------------------add
  //
  // sinks (inputs) are named with 2 digits: the first is the branch it is on
  // (0 or 1) and the second is its position from the start of the path.
  //
  //
  // What are we testing?
  //
  // We expect sinks which appear early in the chain to be unwound first,
  // because of the tie-braker which uses the longest path to a terminal op.
  //

  const auto sink01 = getNext(getSink("in00", 230.), getSink("in01"), "01");
  const auto sink02 = getNext(sink01, getSink("in02"), "02");
  const auto sink03 = getNext(sink02, getSink("in03"), "03");
  const auto sink04 = getNext(sink03, getSink("in04"), "04");

  const auto sink11 = getNext(getSink("in10", 220.), getSink("in11"), "11");
  const auto sink12 = getNext(sink11, getSink("in12"), "12");
  const auto sink13 = getNext(sink12, getSink("in13"), "13");
  const auto sink14 = getNext(sink13, getSink("in14"), "14");

  getNext(sink04, sink14, "finale");

  auto soln = Solution(g, Algo::Greedy0);

  // names are of the form in_{branch#}_{index from start}.
  //                       || |          |
  //                       01 2          3
  //
  auto getIndexFromStart = [](const std::string &s) {
    return static_cast<int>(s.at(3) - 48);
  };

  std::vector<int> indicesFromStart;
  for (auto x : soln.barriersToSinks()) {
    indicesFromStart.push_back(getIndexFromStart(g.getName(x.dst().opId())));
  }

  for (uint64_t i = 1; i < indicesFromStart.size(); ++i) {
    if (indicesFromStart[i - 1] > indicesFromStart[i]) {
      throw poprithms::test::error(
          "Expected the 'indices from start' to be in ascending order, due "
          "the tie-breaking on longest path to terminal nodes.");
    }
  }
}

} // namespace

int main() {
  testOrder0();
  return 0;
}
