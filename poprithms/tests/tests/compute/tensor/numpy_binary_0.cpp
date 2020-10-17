// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <numeric>

#include <poprithms/compute/host/error.hpp>
#include <poprithms/compute/host/tensor.hpp>

namespace {

using namespace poprithms::compute::host;

void test1() {
  const auto aNon = Tensor::arangeInt32(-1, 23, 2).slice_({1}, {11});
  const auto bNon = Tensor::arangeInt32(21, 9, -1).slice_({1}, {11});

  // Cannot currently inplace binary when aliases
  bool didCatch = false;
  try {
    const auto foo = aNon.expand_({10, 10}).add_(bNon.reshape({1, 10}));
  } catch (const poprithms::error::error &) {
    didCatch = true;
  }
  if (!didCatch) {
    // Plan is to support certain cases of this.
    throw error("Failed to catch case of inplace alias add");
  }
}

void test2() {
  const auto aNon =
      Tensor::arangeInt32(-1, 23, 2).slice_({1}, {11}).reshape_({1, 1, 10});
  const auto bNon =
      Tensor::arangeInt32(21, 9, -1).slice_({1}, {11}).reshape_({2, 5, 1});

  const auto x0 = aNon + bNon - aNon - bNon;
  if (!x0.abs().allZero()) {
    throw error("Error in test 0 (+ and -)");
  }

  const auto x1 = aNon / bNon - aNon / bNon;
  if (!x1.abs().allZero()) {
    throw error("Error in test 0 (/ and -)");
  }

  const auto x2 = aNon * bNon - bNon * aNon;
  if (!x1.abs().allZero()) {
    throw error("Error in test 0 (* and -)");
  }

  const auto x3 = (aNon < bNon).toInt32() + (aNon >= bNon).toInt32() -
                  Tensor::boolean(true).toInt32();
  if (!x3.abs().allZero()) {
    throw error("Error in test 0 (< and >=)");
  }

  const auto x4 = (aNon <= bNon).toInt16() + (aNon > bNon).toInt16() -
                  Tensor::boolean(true).toInt16();
  if (!x4.abs().allZero()) {
    throw error("Error in test 0 (<= and >)");
  }
}

void test3() {
  const auto aCon = Tensor::arangeInt32(0, 11, 1);
  const auto aNon = aCon.slice_({1}, {11});
  const auto foo  = aNon.add_(aNon);
  aCon.assertAllEquivalent(Tensor::arangeInt32(0, 22, 2));
}

void test4() {
  // Check that you can do binary inplace Ops when both lhs and rhs are
  // ViewDatas.
  const auto x  = Tensor::arangeInt16(0, 36, 1).reshape({6, 6});
  const auto s0 = x.slice_({1, 2}, {4, 5});
  const auto s1 = x.slice_({2, 1}, {5, 4});
  const auto x0 = s0.add(s1);
  const auto x1 = s0.add_(s1);
  x0.assertAllEquivalent(x1);
  x1.assertAllEquivalent(s0);
}

} // namespace

int main() {
  test1();
  test2();
  test3();
  test4();
}
