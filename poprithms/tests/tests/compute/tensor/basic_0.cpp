// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <numeric>

#include <poprithms/compute/host/error.hpp>
#include <poprithms/compute/host/tensor.hpp>

namespace {
using namespace poprithms::compute::host;

void testZero() {
  const auto t = Tensor::boolean({2}, {true, false});
  if (t.allZero() || t.allNonZero()) {
    throw error("t contains a mix of true and false");
  }
  const auto tTrue  = Tensor::boolean({2}, {true, true});
  const auto tFalse = Tensor::boolean({2}, {false, false});
  if (!tTrue.allNonZero() || !tFalse.allZero()) {
    throw error("tTrue are all true, tFalse are all false");
  }

  const auto t0 = Tensor::float64({3}, {0., 0., 0.});
  if (t0.allNonZero() || !t0.allZero()) {
    throw error("t0 is all zeros");
  }
}

void testAllClose() {

  // testing
  // absolute(a - b) <= (atol + rtol * absolute(b)).

  Tensor t0        = Tensor::float32({1}, {10.0});
  Tensor t1        = Tensor::float32({1}, {11.0});
  const auto atol0 = 0.2;
  const auto atol1 = 1.5;
  const auto rtol0 = 0.02;
  const auto rtol1 = 0.15;

  if (!t0.allClose(t1, rtol1, atol1)) {
    throw error("should be close atol1 and rtol1");
  }

  if (!t0.allClose(t1, rtol0, atol1)) {
    throw error("should be close with atol0 and rtol1");
  }

  if (!t0.allClose(t1, rtol1, atol0)) {
    throw error("should be close with atol1 and rtol0");
  }

  if (t0.allClose(t1, rtol0, atol0)) {
    throw error("shouldn't be close with atol0 and rtol0");
  }

  t0.assertAllEquivalent(t0);
}

void testIdenticalTo() {

  const auto t0 = Tensor::int32(1);
  const auto t1 = Tensor::int32(1);
  if (t0.identicalTo(t1) || !t0.identicalTo(t0)) {
    throw error("Error in indenticalTo test");
  }
}

void testIsOrigin() {
  const auto t0 = Tensor::int32({2, 2}, {2, 3, 4, 5});
  auto t1       = t0.reshape_({4}).slice_({1}, {3});
  if (t0.implIsView() || t1.implIsOrigin()) {
    throw error("Error in testIsOrigin");
  }
}

void testAtSlice0() {
  const auto t0 =
      Tensor::arangeInt32(0, 4, 1).reshape({4, 1, 1}).expand({4, 3, 2});
  t0.at(1).assertAllEquivalent(Tensor::int32(1).expand({3, 2}));
  t0.at(2).assertAllEquivalent(Tensor::int32(2).expand({3, 2}));

  // Inplace slice creates reference to sliced tensor
  t0.at_(1).zeroAll_();
  t0.at(1).assertAllEquivalent(Tensor::int32(0).expand({3, 2}));
}

void testAtSlice1() {
  const auto t0 =
      Tensor::arangeInt32(0, 4, 1).reshape({4, 1, 1}).expand({4, 3, 2});

  // Slice on non-negative integers only:
  {
    bool caught{false};
    try {
      t0.at_(Tensor::int32(-1));
    } catch (const poprithms::error::error &) {
      caught = true;
    }
    if (!caught) {
      throw error(
          "Failed to catch error of slicing with at(.) on negative index");
    }
  }

  // Slice on scalars only:
  {
    bool caught{false};
    try {
      t0.at_(Tensor::unsigned32({0, 2, 3}, {}));
    } catch (const poprithms::error::error &) {
      caught = true;
    }
    if (!caught) {
      throw error(
          "Failed to catch error when slicing with at(.) on non-scalar");
    }
  }
}

void testSlice0() {
  auto a = Tensor::arangeInt32(0, 2 * 3 * 4, 1).reshape({2, 3, 4});
  auto b = a.slice({1, 0, 0}, {2, 3, 1});
  auto c = a.slice(Dimensions({0, 2}), {1, 0}, {2, 1});
  b.assertAllEquivalent(c);
}

void testAccumulate0() {
  Tensors ts;
  for (uint64_t i = 0; i < 10; ++i) {
    ts.push_back(Tensor::unsigned64(i).expand({3, 2}));
  }
  auto out = Tensor::accumulate_(ts, CommutativeOp::Sum);
  out.assertAllEquivalent(ts[0]);
  out.assertAllEquivalent(Tensor::unsigned64(45).expand_({3, 2}));
}

} // namespace

int main() {

  testZero();
  testAllClose();
  testIdenticalTo();
  testIsOrigin();
  testAtSlice0();
  testAtSlice1();
  testSlice0();
  testAccumulate0();

  return 0;
}
