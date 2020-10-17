// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <numeric>

#include <poprithms/compute/host/error.hpp>
#include <poprithms/compute/host/tensor.hpp>

namespace {

using namespace poprithms::compute::host;

// Test that values are preserved, and no obvious errors occur.
void test0() {
  const Shape shape({2, 3, 5});
  const auto N  = static_cast<int32_t>(shape.nelms_u64());
  const auto t0 = Tensor::arangeInt32(0, N, 1);
  const auto t1 = t0.reshape(shape).flatten_();
  const auto t2 = t0.reshape_(shape).flatten();
  const auto t3 = t0.reshape(shape).flatten();
  const auto t4 = t0.reshape_(shape).flatten_();

  t0.assertAllEquivalent(t1);
  t0.assertAllEquivalent(t2);
  t0.assertAllEquivalent(t3);
  t0.assertAllEquivalent(t4);
}

// Test that reshape_ creates aliases.
void test1() {
  const Shape shape({2, 3});
  const auto N  = static_cast<int32_t>(shape.nelms_u64());
  const auto t0 = Tensor::arangeInt32(0, N, 1);
  // 0 1 2
  // 3 4 5

  const auto t1 = t0.reshape_({3, 2})
                      .mul_(Tensor::int32(10))
                      .reshape_({2, 3})
                      .flatten()
                      .mul_(Tensor::int32(20));
  t0.assertAllEquivalent(Tensor::arangeInt32(0, 10 * N, 10));
}

// Test that reshape_ works for non-contiguous.
void test2() {
  const auto t0 = Tensor::arangeInt8(0, 10, 1).reshape_({1, 10});
  const auto t1 = Tensor::arangeInt8(50, 60, 1).reshape_({1, 10});
  const auto t2 = Tensor::arangeInt8(100, 110, 1).reshape_({1, 10});

  // 0   1   2   3   ...
  // 50  51  52  53  ...
  // 100 101 102 103 ...
  //
  const auto t3 = concat_({t0, t1, t2}, 0);
  const auto t4 = t3.slice_({0, 0}, {3, 5});
  const auto t5 = t4.mul_(Tensor::int8(4));
  t0.flatten_().slice_({0}, {5}).assertAllEquivalent(
      Tensor::arangeInt8(0, 20, 4));
}

} // namespace

int main() {

  test0();
  test1();
  test2();

  return 0;
}
