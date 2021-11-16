// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

namespace {

using namespace poprithms::memory::inplace;

void testBasic0() {
  Graph g;

  // 4x4 tensor
  const auto x1 = Tensor::variable(g, {4, 4});
  if (!x1.contains(x1)) {
    throw poprithms::test::error("x1 does contain x1");
  }

  // 3x3 tensor
  const auto x2 = x1.slice({0, 0}, {3, 3});
  if (!x1.contains(x2)) {
    throw poprithms::test::error("x1 does x2, contain a slice of itself");
  }

  if (x2.contains(x1)) {
    throw poprithms::test::error(
        "x2 does not contain x1, from which it is sliced");
  }

  const auto x3 = Tensor::concat({x2, x2, x2}, 0);
  if (!x3.contains(x2)) {
    throw poprithms::test::error(
        "x3 contains x2, a concatenation of itself 3 times");
  }
  if (!x2.contains(x3)) {
    throw poprithms::test::error(
        "x2 contains x3, it is derived 100 percent from it");
  }

  if (x3.contains(x1)) {
    throw poprithms::test::error("x3 does not contain x1");
  }
}

void assertContains(const Tensor &super, const Tensor &sub) {
  if (!super.contains(sub)) {
    std::ostringstream oss;
    oss << "Failure in assertContains(super=" << super << ", sub=" << sub
        << "). ";
    oss << "'super' determined to not contain 'sub'.";
    throw poprithms::test::error(oss.str());
  }
}

void assertNotContains(const Tensor &super, const Tensor &sub) {
  if (super.contains(sub)) {
    std::ostringstream oss;
    oss << "Failure in assertNotContains(super=" << super << ", sub=" << sub
        << "). ";
    oss << "'super' determined to contain 'sub'.";
    throw poprithms::test::error(oss.str());
  }
}

void testBasic1() {

  Graph g;

  const auto t0 = Tensor::variable(g, {10, 10});
  const auto t1 = Tensor::constant(g, {10, 10});
  const auto t2 = Tensor::variable(g, {10, 10});
  Tensor::constant(g, {10, 10});

  auto t01  = Tensor::concat({t0, t1}, 0).reverse(0);
  auto t12  = Tensor::concat({t1, t2}, 1).dimShuffle({{1, 0}});
  auto t012 = Tensor::concat({t01, t12}, 0);

  assertNotContains(t0, t1);
  assertNotContains(t01, t2);
  assertNotContains(t01, t12);
  assertNotContains(t01, t012);

  assertContains(t01, t0);
  assertContains(t12, t2);
  assertContains(t012, t01);
  assertContains(t012, t12);

  auto output0 = t01.subSample(Stride(2), Dimension(0));
  auto output1 =
      t01.slice({1, 0}, {20, 10}).subSample(Stride(2), Dimension(0));
  auto outputCat = Tensor::concat({output0, output1}, 1);
  assertContains(outputCat, t0);
  assertContains(outputCat, t1);
  assertNotContains(output0, t0);
  assertNotContains(output0, t1);
}

void testBasic2(bool openGate, uint64_t stride) {

  Graph g;
  const auto t0 = Tensor::variable(g, {37, 11, 3});

  const auto output =
      t0.dimShuffle({{2, 1, 0}})
          .modify()
          .reverse(2)
          .flatten()
          .pad({3}, {5}, ConstantPadding::Yes, BroadcastPadding::No)
          .aliasGate(openGate)
          .modify()
          .subSample(Stride(stride), Dimension(0));

  // Because of the padding, t0 should never contain output.
  assertNotContains(t0, output);

  // Only when the gate is open and the subsampling is disabled can the output
  // contain the input.
  if (openGate && (stride == 1)) {
    assertContains(output, t0);
  } else {
    assertNotContains(output, t0);
  }
}

} // namespace

int main() {
  testBasic0();
  testBasic1();
  testBasic2(true, 1);
  testBasic2(true, 2);
  testBasic2(false, 1);
  testBasic2(false, 2);
  return 0;
}
