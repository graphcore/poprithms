// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <limits>
#include <numeric>

#include <poprithms/compute/host/error.hpp>
#include <poprithms/compute/host/tensor.hpp>

namespace {
using namespace poprithms::compute::host;

void testSumReduce0() {

  // [[ 1 2 3 4 5  ]
  //  [ 6 7 8 9 10 ]]
  const auto a = Tensor::arangeInt32(1, 11, 1).reshape({2, 5});

  // [[ 15 ]
  //  [ 40 ]]
  a.reduceSum(Shape({2, 1}))
      .assertAllEquivalent(Tensor::int32({2, 1}, {15, 40}));

  // [[ 7 9 11 13 15 ]]
  a.reduceSum(Shape({1, 5}))
      .assertAllEquivalent(Tensor::int32({1, 5}, {7, 9, 11, 13, 15}));

  // [[ 55 ]]
  a.reduceSum(Shape({1, 1})).assertAllEquivalent(Tensor::int32({1, 1}, {55}));

  // [ 55 ]
  a.reduceSum(Shape({1})).assertAllEquivalent(Tensor::int32({1}, {55}));

  // scalar(55)
  a.reduceSum(Shape({})).assertAllEquivalent(Tensor::int32({}, {55}));
}

void testProdReduce0() {

  //
  // [[[ 1 2 ]
  //   [ 3 4 ]]
  //  [[ 5 6 ]
  //   [ 7 8 ]]]
  //
  //
  Tensor::arangeInt64(1, 9, 1)
      .reshape({2, 2, 2})
      .reduceProduct({1, 2, 1})
      .assertAllEquivalent(
          Tensor::int64({1, 2, 1}, {1 * 2 * 5 * 6, 3 * 4 * 7 * 8}));

  // true * true = true
  // true * false = false
  // false * false = false
  Tensor::boolean({3, 2}, {true, true, true, false, false, false})
      .reduceProduct({3, 1})
      .assertAllEquivalent(Tensor::boolean({3, 1}, {true, false, false}));
}

void testMinMaxReduce0Float32() {

  const auto uniran = Tensor::uniformFloat32(-1, 1, {2, 3, 4, 5, 6}, 1011);
  const auto tMin   = uniran.reduceMin({});
  const auto tMax   = uniran.reduceMax({});

  const auto v = uniran.getFloat32Vector();
  auto min_    = std::accumulate(v.cbegin(),
                              v.cend(),
                              std::numeric_limits<float>::max(),
                              [](auto a, auto b) { return std::min(a, b); });

  auto max_ = std::accumulate(v.cbegin(),
                              v.cend(),
                              std::numeric_limits<float>::lowest(),
                              [](auto a, auto b) { return std::max(a, b); });

  tMin.assertAllEquivalent(Tensor::float32(min_));
  tMax.assertAllEquivalent(Tensor::float32(max_));
}

void testMinMaxReduce0Float16() {
  const auto uniran = Tensor::arangeFloat16(-1, 1, 0.5);
  const auto tMin   = uniran.reduceMin({});
  const auto tMax   = uniran.reduceMax({});
  tMin.assertAllEquivalent(Tensor::float16(-1.0));
  tMax.assertAllEquivalent(Tensor::float16(0.5));
}

} // namespace

int main() {
  testSumReduce0();
  testProdReduce0();
  testMinMaxReduce0Float32();
  testMinMaxReduce0Float16();
  return 0;
}
