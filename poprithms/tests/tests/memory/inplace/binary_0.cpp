// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <random>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>

namespace {
using namespace poprithms::memory::inplace;

void testBadShape() {
  Graph g;
  const auto a = g.variable({3, 1});
  const auto b = g.variable({3, 3});
  const auto c = g.binary(a, b, AliasType::outplace());
  bool caught{false};
  try {
    g.tryInplace({c, AliasType::binary0()}, CheckParallelWriteable::Yes);
  } catch (const poprithms::error::error &) {
    caught = true;
  }
  if (!caught) {
    throw error("Failed to catch error of inplacing on broadcast arg");
  }
}

void testNoConst() {
  Graph g;
  const auto a = g.constant({3});
  const auto b = g.variable({3});
  const auto c = g.binary(a, b, AliasType::outplace());
  g.tryInplaces({{c, AliasType::binary0()}, {c, AliasType::binary1()}},
                CheckParallelWriteable::Yes);
  auto alis = g.allAliases(c);
  std::sort(alis.begin(), alis.end());
  if (alis != TensorIds{b, c}) {
    throw error("Incorrect aliases after inplacing");
  }
}

void testMultiplePossibilities() {
  Graph g;
  const auto a = g.variable({3});
  const auto b = g.variable({3});
  const auto c = g.binary(a, b, AliasType::outplace());
  // Both valid inplacings, but only the first one should be applied.
  g.tryInplaces({{c, AliasType::binary0()}, {c, AliasType::binary1()}},
                CheckParallelWriteable::Yes);
  auto alis = g.allAliases(c);
  std::sort(alis.begin(), alis.end());
  if (alis != TensorIds{a, c}) {
    throw error("Incorrect aliases after inplacing");
  }
}

void testChain0() {
  Graph g;
  TensorIds ids{g.variable({7})};
  TensorIds binIds{};
  while (binIds.size() < 6) {
    ids.push_back(g.variable({7}));
    ids.push_back(
        g.binary(*(ids.cend() - 2), ids.back(), AliasType::outplace()));
    binIds.push_back(ids.back());
  }
  std::mt19937_64 generator(1015);
  std::shuffle(binIds.begin(), binIds.end(), generator);

  Proposals proposals;
  proposals.reserve(binIds.size());
  for (auto id : binIds) {
    proposals.push_back({id, AliasType::binary0()});
  }

  g.tryInplaces(proposals, CheckParallelWriteable::Yes);
  std::cout << g << std::endl;

  for (auto id : ids) {
    if (g.nInTensors(id.opId()) != 0 &&
        g.aliasType(id) == AliasType::outplace()) {
      throw error("Expected all binary ops to be inplaced");
    }
  }
}

} // namespace

int main() {
  testBadShape();
  testNoConst();
  testMultiplePossibilities();
  testChain0();
}
