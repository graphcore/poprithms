// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <random>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

namespace {
using namespace poprithms::memory::inplace;

void testBadShape() {
  Graph g;
  const auto a = Tensor::variable(g, {3, 1});
  const auto b = Tensor::variable(g, {3, 3});
  const auto c = Tensor::aliasGate({a, b});
  bool caught{false};
  try {
    g.tryOpening(
        {c, 0}, CheckParallelWriteable::Yes, AllowMultiGateAlias::No);
  } catch (const poprithms::error::error &) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error(
        "Failed to catch error of inplacing on broadcast arg");
  }
}

void testNoConst() {
  Graph g;
  //
  //
  // a.
  //   \.
  //     aliasGate -> unary
  //   /.
  // b
  const auto a = Tensor::constant(g, {3});
  const auto b = Tensor::variable(g, {3});
  const auto c = Tensor::aliasGate({a, b});
  c.modify();
  g.tryOpenings(
      {{c, 0}, {c, 1}}, CheckParallelWriteable::Yes, AllowMultiGateAlias::No);
  auto alis = c.allAliases();
  if (std::find(alis.cbegin(), alis.cend(), a) != alis.cend()) {
    throw poprithms::test::error(
        "Expected a to NOT be aliased to c, as it is constant");
  }
  if (std::find(alis.cbegin(), alis.cend(), b) == alis.cend()) {
    throw poprithms::test::error(
        "Expected b TO be aliased to c, as it is not constant");
  }
}

void testSkittyAccl() {

  Graph g;
  //
  // modeling the program:
  //
  //    a = var()
  //    b = var()
  //    c = var()
  //    d = a + b
  //    e = b + c
  //    dMod = f.relu_()
  //    eMod = e.relu_()
  //
  // a ----+
  //       |
  //     aliasGate --- d ---> modify
  //       |
  // b ----+
  //       |
  //     aliasGate --- e ---> modify
  //       |
  // c ----+
  //
  const auto a    = Tensor::variable(g, {3});
  const auto b    = Tensor::variable(g, {3});
  const auto c    = Tensor::variable(g, {3});
  const auto d    = Tensor::aliasGate({a, b});
  const auto e    = Tensor::aliasGate({b, c});
  const auto dMod = d.modify();
  e.modify();

  auto g2 = g;

  // try to make
  //   d = b_.add(a)
  //   e = c_.add(b).
  //
  //  This is fine, as long as d is created as only after b has been used
  //  (order of operations need to change).
  //
  auto x = g.tryOpenings(
      {{d, 1}, {e, 1}}, CheckParallelWriteable::Yes, AllowMultiGateAlias::No);
  if (d.aliasGateIsClosed() || e.aliasGateIsClosed()) {
    throw poprithms::test::error("Expected both gates to be opened");
  }

  {
    // verify that when d is inplaced to 'd = b_.add(a)', there are
    // constraints inserted:
    auto x2 = g2.tryOpeningPartial(
        {d, 1}, CheckParallelWriteable::Yes, AllowMultiGateAlias::No);
    auto cs = x2.constraints();
    Constraint expected{e.opId(), dMod.opId()};

    if (std::find(cs.begin(), cs.end(), expected) == cs.end()) {
      throw poprithms::test::error("Expected constraint is not present");
    }
  }
}

void testMultiplePossibilities() {
  Graph g;
  const auto a = Tensor::variable(g, {3});
  const auto b = Tensor::variable(g, {3});
  const auto c = Tensor::aliasGate({a, b});
  // Both valid inplacings, but only the first one should be applied.
  g.tryOpenings(
      {{c, 0}, {c, 1}}, CheckParallelWriteable::Yes, AllowMultiGateAlias::No);
  auto alis = c.allAliases();
  std::sort(alis.begin(), alis.end());
  if (alis != Tensors{a, c}) {
    throw poprithms::test::error("Incorrect aliases after inplacing");
  }
}

void testChain0() {
  Graph g;
  Tensors all{Tensor::variable(g, {7})};
  Tensors aliasGates{};
  while (aliasGates.size() < 6) {
    all.push_back(Tensor::variable(g, {7}));
    auto m   = Tensor::aliasGate({*(all.cend() - 2), all.back()});
    auto mun = m.modify();
    all.push_back(mun);
    aliasGates.push_back(m);
  }
  std::mt19937_64 generator(1015);
  std::shuffle(aliasGates.begin(), aliasGates.end(), generator);
  g.tryOpenings0(Tensor::opIds(aliasGates),
                 CheckParallelWriteable::Yes,
                 AllowMultiGateAlias::No);
  for (auto m : aliasGates) {
    if (m.aliasGateIsClosed()) {
      throw poprithms::test::error(
          "Expected all aliasGate ops to be inplaced");
    }
  }
}

} // namespace

int main() {
  testBadShape();
  testNoConst();
  testMultiplePossibilities();
  testChain0();
  testSkittyAccl();
}
