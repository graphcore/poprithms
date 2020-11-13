// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>

namespace {

using namespace poprithms::memory::inplace;

Proposals getProposals0(const std::vector<TensorId> &ids) {
  Proposals ps;
  ps.reserve(ids.size());
  for (const auto &id : ids) {
    ps.push_back({id, AliasType::allInplace()});
  }
  return ps;
}

void testUnaryChainBase(const Graph &g0, const TensorIds &order) {
  auto g = g0;
  const auto statuses =
      g.tryInplaces(getProposals0(order), CheckParallelWriteable::Yes);
  for (auto x : order) {
    if (g.aliasType(x.opId()) != AliasType::allInplace()) {
      std::ostringstream oss;
      oss << "With order = " << order << " failed to inplace all";
      throw error(oss.str());
    }
  }
  for (auto s : statuses) {
    if (s != InplaceStatus::Valid) {
      std::ostringstream oss;
      oss << "With order = " << order << " failed to apply all";
      throw error(oss.str());
    }
  }
}

void testUnaryChain() {
  Graph g;
  auto x0 = g.variable({4, 4});
  auto x1 = g.unary(x0, AliasType::outplace());
  auto x2 = g.unary(x1, AliasType::outplace());
  auto x3 = g.unary(x2, AliasType::outplace());
  testUnaryChainBase(g, {x3, x1, x2});
  testUnaryChainBase(g, {x2, x1, x3});
  testUnaryChainBase(g, {x2, x3, x1});
  testUnaryChainBase(g, {x1, x2, x3});
}

void testUnaryTriFork0Base(const Graph &g0, const TensorIds &order) {
  auto g = g0;
  const auto statuses =
      g.tryInplaces(getProposals0(order), CheckParallelWriteable::Yes);

  if (order.size() != 3) {
    throw error("order must be of size 3 in this test - bad test");
  }

  // We expect only the first proposal to be accepted:
  if (g.aliasType(order[0].opId()) != AliasType::allInplace()) {
    std::ostringstream oss;
    oss << "With order = " << order << ", failed to inplace first";
    throw error(oss.str());
  }

  for (auto i : {1, 2}) {
    if (g.aliasType(order[i].opId()) == AliasType::allInplace()) {
      std::ostringstream oss;
      oss << "With order = " << order << ", inplaced after first";
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
  auto x0 = g.variable({3});
  auto x1 = g.unary(x0, AliasType::outplace());
  auto x2 = g.unary(x0, AliasType::outplace());
  auto x3 = g.unary(x0, AliasType::outplace());

  testUnaryTriFork0Base(g, {x1, x2, x3});
  testUnaryTriFork0Base(g, {x3, x2, x1});
  testUnaryTriFork0Base(g, {x2, x3, x1});
}

void testUnaryTriLongFork0() {

  //     +----- x0 -----+
  //     |      |       |
  //   unary  unary   unary
  //     |      |       |
  //    x1     x3      x5
  //     |      |       |
  //   unary  unary   unary
  //     |      |       |
  //    x2     x4      x6
  //
  //    We expect the first of {x1, x3, x5} to be inplace,
  //                  =====
  //    and all of {x2, x4, x6}.

  Graph g0;
  auto x0 = g0.variable({3});
  TensorIds outs;
  TensorIds forkers;
  for (uint64_t i = 0; i < 3; ++i) {
    for (uint64_t j = 0; j < 2; ++j) {
      if (j % 2 == 0) {
        outs.push_back(g0.unary(x0, AliasType::outplace()));
        forkers.push_back(outs.back());
      } else {
        outs.push_back(g0.unary(outs.back(), AliasType::outplace()));
      }
    }
  }

  auto isForker = [&forkers](TensorId id) {
    return std::find(forkers.cbegin(), forkers.cend(), id) != forkers.cend();
  };

  std::vector<TensorIds> orders{
      outs,                                                   //
      {outs[5], outs[4], outs[3], outs[2], outs[1], outs[0]}, //
      {outs[4], outs[5], outs[2], outs[3], outs[1], outs[0]}, //
      {outs[5], outs[3], outs[3], outs[1], outs[0], outs[4]}, //
  };
  for (auto order : orders) {

    auto g = g0;
    g.tryInplaces(getProposals0(order), CheckParallelWriteable::Yes);

    auto getFirstForker = [&order, &isForker]() -> TensorId {
      for (auto x : order) {
        if (isForker(x)) {
          return x;
        }
      }
      throw error("first forker not found");
    };

    auto firstForker = getFirstForker();
    for (auto x : order) {
      if (isForker(x) && x != firstForker) {
        if (g.aliasType(x.opId()) != AliasType::outplace()) {
          std::ostringstream oss;
          oss << "With order = " << order << ", expected " << x
              << " to be outplace";
          throw error(oss.str());
        }
      } else {
        if (g.aliasType(x.opId()) != AliasType::allInplace()) {
          std::ostringstream oss;
          oss << "With order = " << order << ", expected " << x
              << " to be inplace";
          throw error(oss.str());
        }
      }
    }
  }
}

void testMixedBiFork0Base(const Graph &g0,
                          CheckParallelWriteable obey,
                          const TensorIds &order,
                          const TensorIds &expectedOut) {
  auto g              = g0;
  Proposals proposals = getProposals0(order);
  const auto statuses = g.tryInplaces(proposals, obey);

  auto getBaseString = [&g0, &order, &expectedOut]() {
    std::ostringstream oss;
    oss << "For Initial Graph=" << g0 << ", and order=" << order
        << " expected " << expectedOut << " to be outplace. ";
    return oss.str();
  };

  for (auto tId : order) {
    if (std::find(expectedOut.cbegin(), expectedOut.cend(), tId) !=
        expectedOut.cend()) {
      if (g.aliasType(tId.opId()) != AliasType::outplace()) {
        std::ostringstream oss;
        oss << getBaseString() << "\nFailed, as " << tId
            << " is not outplace. "
            << "Results were " << statuses;
        throw error(oss.str());
      }
    } else {
      if (g.aliasType(tId.opId()) == AliasType::outplace()) {
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
  //    rsh    rev  // view change copies
  //     |      |
  //    x0     x1   // unary modfier copies
  //      \   /.
  //       cat      // concatenation copy
  //
  auto alloc = g.variable({7});
  auto rsh   = g.reshape(alloc, AliasType::outplace(), {7});
  auto x0    = g.unary(rsh, AliasType::outplace());
  auto rev   = g.reverse(alloc, AliasType::outplace(), {0});
  auto x1    = g.unary(rev, AliasType::outplace());
  auto cat   = g.concat({x0, x1}, AliasType::outplace(), 0);

  const auto T = CheckParallelWriteable::Yes;
  const auto F = CheckParallelWriteable::No;
  testMixedBiFork0Base(g, T, {x1, x0, rsh, rev, cat}, {rev});
  testMixedBiFork0Base(g, F, {rev, rsh, cat, x0, x1}, {x1});
  testMixedBiFork0Base(g, T, {cat, x0, rsh, x1, rev}, {rev});
  testMixedBiFork0Base(g, F, {cat, x1, x0, rev, rsh}, {rsh});
}

void testConstraint0() {
  Graph g;
  const auto alloc = g.variable({3});
  const auto x0    = g.unary(alloc, AliasType::outplace());
  const auto x1    = g.unary(alloc, AliasType::outplace());

  //
  //      alloc
  //      /.   \.
  //    x0  <-  x1
  //
  g.constraint(x1, x0);

  // The attempt to inplace x1 fails, as it
  // is constrained to be before x0.
  TensorIds order{x1, x0};
  g.tryInplaces(getProposals0(order), CheckParallelWriteable::Yes);
  if (g.aliasType(x1) != AliasType::outplace()) {
    throw error(
        "Failed to inplace correctly with constraint - x0 not outplace");
  }

  if (g.aliasType(x0) != AliasType::allInplace()) {
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
