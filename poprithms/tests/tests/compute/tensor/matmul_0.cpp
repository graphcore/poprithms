// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <numeric>

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>

namespace {

using namespace poprithms::compute::host;

void test0() {

  // 1111     11111
  // 1111  x  11111
  // 1111     11111
  //          11111

  Tensor::int32(1)
      .expand({3, 4})
      .matmul(Tensor::int32(1).expand({4, 5}))
      .assertAllEquivalent(Tensor::int32(4).expand({3, 5}));
}

void test1() {
  Tensor::int32(1)
      .expand({2, 1, 3, 4})
      .matmul(Tensor::int32(1).expand({1, 2, 4, 5}))
      .assertAllEquivalent(Tensor::int32(4).expand({2, 2, 3, 5}));
}

void test2() {

  // 5 2 4    1
  // 1 6 2    2
  //          3

  const auto a = Tensor::float32({2, 3}, {5, 2, 4, 1, 6, 2});
  const auto b = Tensor::float32({3}, {1, 2, 3});
  const auto c = Tensor::float32({2}, {5 + 4 + 12, 1 + 12 + 6});
  c.assertAllEquivalent(a.matmul(b));
}

void test3() {

  //  1 2 3   4 5
  //          6 7
  //          8 9

  const auto a = Tensor::float64({3}, {1, 2, 3});
  const auto b = Tensor::float64({3, 2}, {4, 5, 6, 7, 8, 9});
  a.matmul(b).assertAllEquivalent(
      Tensor::float64({2}, {1 * 4 + 2 * 6 + 3 * 8, 1 * 5 + 2 * 7 + 3 * 9}));
}

void test4() {
  const auto a = Tensor::float64({1, 3}, {1, 2, 3});
  const auto b =
      Tensor::float64({3, 2}, {4, 5, 6, 7, 8, 9}).expand({1, 5, 1, 3, 2});
  const auto c = a.matmul(b);
  if (c.shape() != Shape({1, 5, 1, 1, 2})) {
    throw poprithms::test::error("Incorrect output Shape in test4");
  }
}

void test5() {
  const auto a = Tensor::unsigned16(1).expand({2, 3, 4, 1, 1});
  const auto b = Tensor::unsigned16(1).expand({1, 5});
  a.matmul(b).assertAllEquivalent(
      Tensor::unsigned16(1).expand({2, 3, 4, 1, 5}));
}

void test6() {

  // AB = (B^T.A^T)^T
  //
  const auto a = Tensor::uniformFloat16(0, 5, {2, 3}, 1011);
  const auto b = Tensor::uniformFloat16(0, 5, {3, 4}, 10111);

  const auto c = a.matmul(b);
  const auto d = b.dimShuffle({{1, 0}})
                     .matmul(a.dimShuffle({{1, 0}}))
                     .dimShuffle({{1, 0}});

  const auto relTol = 0.0;
  const auto absTol = 1e-1;
  c.assertAllClose(d, relTol, absTol);
}

void test7() {
  // 2 x 6 x 5
  const auto a = Tensor::int32(1).expand({2, 12, 10}).subSample_({1, 2, 2});

  // 2 x 5 x 4
  const auto b = Tensor::int32(1).expand({2, 10, 28}).subSample_({1, 2, 7});

  // matmul, where neither of the tensors is an origin tensor.
  a.matmul(b).assertAllEquivalent(Tensor::int32(5).expand({2, 6, 4}));
}

} // namespace

int main() {
  test0();
  test1();
  test2();
  test3();
  test4();
  test5();
  test6();
  test7();
  return 0;
}
