// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

} // namespace

int main() {

  testZero();
  testAllClose();
  testIdenticalTo();
  testIsOrigin();

  return 0;
}
