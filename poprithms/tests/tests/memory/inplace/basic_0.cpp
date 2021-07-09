// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

namespace {

using namespace poprithms::memory::inplace;

// Check that all AliasGatees are open.
void testUnaryChainBase(Graph g, const TensorIds &idOrder) {

  const auto order    = Tensor::tensors(g, idOrder);
  const auto statuses = g.tryOpenings0(Tensor::tensorIds(order),
                                       CheckParallelWriteable::Yes,
                                       AllowMultiGateAlias::No);
  for (auto t : order) {
    if (t.aliasGateIsClosed()) {
      std::ostringstream oss;
      oss << "In this test, "
          << " which consists of a simple chain of "
          << " unary and aliasGate, "
          << " all aliasGates should be opened. "
          << "With order = " << order
          << ", failed to open all, statuses = " << statuses;
      throw error(oss.str());
    }
  }
  for (auto s : statuses) {
    if (s != OpeningStatus::Valid) {
      std::ostringstream oss;
      oss << "With order = " << order
          << " not all statuses are Valid, they should be";
      throw error(oss.str());
    }
  }
}

void testUnaryChain() {
  Graph g;
  auto x1 = Tensor::variable(g, {4, 4}).closedAliasGate();
  auto x2 = x1.modify().closedAliasGate();
  auto x3 = x2.modify().closedAliasGate();
  x3.modify();

  testUnaryChainBase(g, Tensor::tensorIds({x3, x1, x2}));
  testUnaryChainBase(g, Tensor::tensorIds({x2, x1, x3}));
  testUnaryChainBase(g, Tensor::tensorIds({x2, x3, x1}));
  testUnaryChainBase(g, Tensor::tensorIds({x1, x2, x3}));
}

void testUnaryTriFork0Base(Graph g, const TensorIds &idOrder) {
  const auto order    = Tensor::tensors(g, idOrder);
  const auto statuses = g.tryOpenings0(
      idOrder, CheckParallelWriteable::Yes, AllowMultiGateAlias::No);

  if (order.size() != 3) {
    throw error("order must be of size 3 in this test - bad test");
  }

  // We expect only the first proposal to be accepted:
  if (order[0].aliasGateIsClosed()) {
    std::ostringstream oss;
    oss << "With order = " << order << ", failed to inplace first";
    throw error(oss.str());
  }

  for (auto i : {1, 2}) {
    if (order[i].aliasGateIsOpen()) {
      std::ostringstream oss;
      oss << "With order = " << order << ", incorrectly inplaced non-first";
      throw error(oss.str());
    }
  }
}

void testUnaryTriFork0() {

  //     +----- x0 -----+
  //     |      |       |
  //   unary  unary   unary
  //     |      |       |
  //    x1     x2      x3
  Graph g;
  auto x0 = Tensor::variable(g, {3});
  auto x1 = x0.closedAliasGate();
  auto x2 = x0.closedAliasGate();
  auto x3 = x0.closedAliasGate();
  x1.modify();
  x2.modify();
  x3.modify();

  testUnaryTriFork0Base(g, Tensor::tensorIds({x1, x2, x3}));
  testUnaryTriFork0Base(g, Tensor::tensorIds({x3, x2, x1}));
  testUnaryTriFork0Base(g, Tensor::tensorIds({x2, x3, x1}));
}

void testUnaryTriLongFork0() {

  //     +----- x0 -----+
  //     |      |       |
  //    aliasGate    aliasGate     aliasGate
  //     |      |       |
  //   unary  unary   unary
  //     |      |       |
  //    x1     x3      x5
  //     |      |       |
  //    aliasGate    aliasGate     aliasGate
  //     |      |       |
  //   unary  unary   unary
  //     |      |       |
  //    x2     x4      x6
  //
  //    We expect the first of {x1, x3, x5} to be inplace,
  //                  =====
  //    and all of {x2, x4, x6}.

  Graph g0_;
  auto x0 = Tensor::variable(g0_, {3});
  TensorIds forkers;
  TensorIds outs;
  for (uint64_t i = 0; i < 3; ++i) {
    auto t = x0;
    for (uint64_t j = 0; j < 2; ++j) {
      const auto m = t.closedAliasGate();
      outs.push_back(m.id());
      if (j % 2 == 0) {
        forkers.push_back(m.id());
      }
      t = m.modify();
    }
  }

  const auto isForker = [&forkers](auto id) {
    return std::find(forkers.cbegin(), forkers.cend(), id) != forkers.cend();
  };

  std::vector<std::vector<uint64_t>> orders{{0, 1, 2, 3, 4, 5},
                                            {5, 4, 3, 2, 1, 0},
                                            {4, 5, 2, 3, 1, 0},
                                            {5, 3, 2, 1, 0, 4}};
  for (auto order : orders) {
    TensorIds aliasGateOrder;
    aliasGateOrder.reserve(6);
    for (auto i : order) {
      aliasGateOrder.push_back(outs[i]);
    }

    auto g = g0_;
    g.tryOpenings0(
        aliasGateOrder, CheckParallelWriteable::Yes, AllowMultiGateAlias::No);

    auto firstForker = [&outs, &order, &isForker]() {
      for (auto x : order) {
        if (isForker(outs[x])) {
          return outs[x];
        }
      }
      throw error("first forker not found");
    }();

    for (auto x : order) {
      const auto id = outs[x];
      if (isForker(id) && id != firstForker) {
        if (g.aliasGateIsOpen(id.opId())) {
          std::ostringstream oss;
          oss << "With order = " << aliasGateOrder << ", expected " << x
              << " to be outplace";
          throw error(oss.str());
        }
      } else {
        if (g.aliasGateIsClosed(id.opId())) {
          std::ostringstream oss;
          oss << "With order = " << aliasGateOrder << ", expected " << x
              << " to be inplace";
          throw error(oss.str());
        }
      }
    }
  }
}

void testMixedBiFork0Base(Graph g,
                          CheckParallelWriteable obey,
                          const TensorIds &idsOrder,
                          const TensorIds &idsExpected) {
  const auto gIn = g;

  const auto order                    = Tensor::tensors(g, idsOrder);
  const auto expectedClosedAliasGates = Tensor::tensors(g, idsExpected);

  const auto statuses =
      g.tryOpenings0(Tensor::tensorIds(order), obey, AllowMultiGateAlias::No);

  auto getBaseString = [&gIn, &order, &expectedClosedAliasGates]() {
    std::ostringstream oss;
    oss << "For Initial Graph=" << gIn << ", and order=" << order
        << " expected only " << expectedClosedAliasGates
        << " to be outplace. ";
    return oss.str();
  };

  for (auto tId : order) {

    if (std::find(expectedClosedAliasGates.cbegin(),
                  expectedClosedAliasGates.cend(),
                  tId) != expectedClosedAliasGates.cend()) {
      if (!tId.aliasGateIsClosed()) {
        std::ostringstream oss;
        oss << getBaseString() << "\nFailed, as " << tId
            << " is not outplace. "
            << "Results were " << statuses;
        throw error(oss.str());
      }
    } else {
      if (tId.aliasGateIsClosed()) {
        std::ostringstream oss;
        oss << getBaseString() << "\nFailed, as " << tId
            << " is outplace. Results were " << statuses;
        throw error(oss.str());
      }
    }
  }
}

void testMixedBiFork0() {
  Graph g;

  //       alloc
  //      /.    \.
  //    rsh    rev    // view change copies
  //     |      |
  //    aliasGate    aliasGate
  //     |      |
  //  unary   unary   // unary modfiers
  //     |      |
  //    aliasGate    aliasGate
  //      \   /.
  //       cat      // concatenation copy
  //        |
  //       aliasGate
  //
  //
  //
  const auto alloc    = Tensor::variable(g, {7}); // .variable({7});
  const auto rsh      = alloc.reshape({7}).closedAliasGate();
  const auto rev      = alloc.reverse(0).closedAliasGate();
  const auto rshUnary = rsh.modify().closedAliasGate();
  const auto revUnary = rev.modify().closedAliasGate();
  const auto cat = Tensor::concat({rshUnary, revUnary}, 0).closedAliasGate();

  for (auto pll : {CheckParallelWriteable::Yes, CheckParallelWriteable::No}) {
    testMixedBiFork0Base(
        g,
        pll,
        Tensor::tensorIds({rshUnary, revUnary, rsh, rev, cat}),
        Tensor::tensorIds({rev}));
    testMixedBiFork0Base(
        g,
        pll,
        Tensor::tensorIds({rsh, rev, cat, rshUnary, revUnary}),
        Tensor::tensorIds({rev}));
    testMixedBiFork0Base(
        g,
        pll,
        Tensor::tensorIds({cat, rshUnary, rsh, revUnary, rev}),
        Tensor::tensorIds({rev}));
    testMixedBiFork0Base(
        g,
        pll,
        Tensor::tensorIds({cat, rev, rshUnary, revUnary, rsh}),
        Tensor::tensorIds({rsh}));
  }
}

void testConstraint0() {
  Graph g;
  const auto alloc       = Tensor::variable(g, {3});
  const auto x0aliasGate = alloc.closedAliasGate();
  x0aliasGate.modify();
  const auto x1aliasGate = alloc.closedAliasGate();
  const auto x11         = x1aliasGate.modify();

  //
  //      alloc
  //      /.   \.
  //    aliasGate    aliasGate
  //     |      |
  //   unary  unary
  //     |      |
  //    x0  <-  x1
  //
  g.constraint(x11.opId(), x0aliasGate.opId());

  // The attempt to inplace x1 fails, as it
  // is constrained to be before x0.
  g.tryOpenings0({x1aliasGate.opId(), x0aliasGate.opId()},
                 CheckParallelWriteable::Yes,
                 AllowMultiGateAlias::No);
  if (x1aliasGate.aliasGateIsOpen()) {
    throw error(
        "Failed to inplace correctly with constraint - x0 not outplace");
  }

  if (x0aliasGate.aliasGateIsClosed()) {
    throw error(
        "Failed to inplace correctly with constraint - x1 not outplace");
  }
}

} // namespace

int main() {
  testMixedBiFork0();
  testUnaryChain();
  testUnaryTriFork0();
  testUnaryTriLongFork0();
  testConstraint0();
  return 0;
}
