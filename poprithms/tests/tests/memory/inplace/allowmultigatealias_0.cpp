// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <random>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

namespace {
using namespace poprithms::memory::inplace;

void testBaseRunner(Graph g,
                    Proposal proposal,
                    bool allowMultiGateAlias,
                    OpeningStatus expected) {

  auto gCopy = g;

  auto observed = g.tryOpening(proposal,
                               CheckParallelWriteable::Yes,
                               allowMultiGateAlias ? AllowMultiGateAlias::Yes
                                                   : AllowMultiGateAlias::No);

  if (observed != expected) {
    std::ostringstream oss;
    oss << "Failed in test of AllowMultiGateAlias. "
        << "The input graph is \n"
        << gCopy << "\nand the proposed AliasGate opening is " << proposal
        << ". This with " << allowMultiGateAlias << ". "
        << "In this situation, the expected OpeningStatus was " << expected
        << ", but the observed OpeningStatus was " << observed << ". ";
    throw error(oss.str());
  }
}

/**
 * (ML) graph:
 * out = a.add(a).
 *
 * Under proposal this would be:
 * a.add_(a)
 *
 * This would create an open alias gate in the poprithms graph where the 2
 * inputs are aliased.
 * */
void test0() {

  Graph g0;
  const auto a          = Tensor::variable(g0, {3, 4});
  const auto aliasGate_ = Tensor::aliasGate({a, a});
  aliasGate_.modify();

  testBaseRunner(g0, {aliasGate_, 0}, false, OpeningStatus::GateMultiInAlias);
  testBaseRunner(g0, {aliasGate_, 0}, true, OpeningStatus::Valid);
}

/**
 * Similar to test0, but now:
 * out = a.slice(bounds0).add(a.slice(bounds1))
 * where a.slice(bounds0) and a.slice(bounds1) intersect.
 *
 * Can the add be inplaced? Same story as test0.
 * */
void test1() {

  Graph g0;
  const auto a          = Tensor::variable(g0, {2, 10});
  const auto leftSlice  = a.slice({0, 0}, {2, 6});
  const auto rightSlice = a.slice({0, 4}, {2, 10});
  const auto gate       = Tensor::aliasGate({leftSlice, rightSlice});
  gate.modify();

  testBaseRunner(g0, {gate, 0}, false, OpeningStatus::GateMultiInAlias);
  testBaseRunner(g0, {gate, 0}, true, OpeningStatus::Valid);
  testBaseRunner(g0, {gate, 1}, false, OpeningStatus::GateMultiInAlias);
  testBaseRunner(g0, {gate, 1}, true, OpeningStatus::Valid);
}

/**
 * Like test1, but now the add is already inplace, and one of the slices is
 * proposed for inplacing.
 * */
void test2() {

  Graph g0;
  const auto a          = Tensor::variable(g0, {2, 10});
  const auto leftSlice  = a.slice({0, 0}, {2, 6});
  const auto sliceGate  = leftSlice.closedAliasGate();
  const auto rightSlice = a.slice({0, 4}, {2, 10});
  Tensor::aliasGate({leftSlice, rightSlice}, 1);

  /**
   * Note that there is no need to put a modify on the ends of the graphs. The
   * logic in poprithms is independent of whether tensors are actually
   * modified.
   * */

  testBaseRunner(g0, {sliceGate, 0}, false, OpeningStatus::GateMultiInAlias);
  testBaseRunner(g0, {sliceGate, 0}, true, OpeningStatus::Valid);
}

/**
 * out = a + a.slice(...).expand()
 * */
void test3() {
  Graph g0;
  const auto a = Tensor::variable(g0, {5, 5});
  const auto b = a.slice({0, 2}, {5, 3}).expand({5, 5});
  auto c       = Tensor::aliasGate({a, b});
  c.modify();
  testBaseRunner(g0, {c, 0}, false, OpeningStatus::GateMultiInAlias);
  testBaseRunner(g0, {c, 0}, true, OpeningStatus::Valid);
  testBaseRunner(g0, {c, 1}, true, OpeningStatus::NotParallelWriteable);
}

/**
 * A slightly more complex example.
 * */
void test4() {

  Graph g0;

  const auto a = Tensor::variable(g0, {24});
  const auto b = Tensor::variable(g0, {1, 1, 1});

  //  contains the element b in it.
  const auto c = Tensor::concat({a.flatten(), b.flatten()}, 0)
                     .reshape({5, 5})
                     .reverse(0)
                     .reverse(1)
                     .dimShuffle({{1, 0}})
                     .slice({0, 0}, {3, 3})
                     .reshape({1, 3, 3});

  // what we will propose opening:
  const auto d = b.expand({1, 3, 1}).closedAliasGate();

  // open alias gate on c.
  Tensor::aliasGate({c, d}, 0);

  testBaseRunner(g0, {d, 0}, false, OpeningStatus::GateMultiInAlias);
  testBaseRunner(g0, {d, 0}, true, OpeningStatus::Valid);
}

} // namespace

int main() {
  test0();
  test1();
  test2();
  test3();
  test4();
}
