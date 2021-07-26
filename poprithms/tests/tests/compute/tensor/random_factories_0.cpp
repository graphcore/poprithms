// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <numeric>
#include <set>

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>

namespace {

using namespace poprithms::compute::host;

void testFloat() {

  const auto a0 = Tensor::uniformFloat32(0., 1.f, {}, 1011);
  const auto a1 = Tensor::uniformFloat32(0., 1.f, {}, 1011);
  const auto b0 = Tensor::uniformFloat32(0., 1.f, {}, 1012);
  const auto x0 = a0.getFloat32(0);
  const auto x1 = a1.getFloat32(0);
  const auto y0 = b0.getFloat32(0);

  if (x0 - x1 != 0.0f) {
    throw poprithms::test::error(
        "Tensors generated with same seed should be identical");
  }

  if (x0 - y0 == 0.0f) {
    throw poprithms::test::error(
        "Tensors generated with different seeds should be the same");
  }

  const auto c0 = Tensor::uniformFloat64(-10., 10., {2, 3, 5, 100}, 1013);
  if (c0.nelms_u64() != 2 * 3 * 5 * 100) {
    throw poprithms::test::error(
        "Incorrect number of elements in random Tensor");
  }
  auto vals = c0.getFloat64Vector();
  std::sort(vals.begin(), vals.end());
  if (vals[0] > -6. || vals.back() < 6.) {
    throw poprithms::test::error(
        "Statistical anomoly? Values probably don't follow uniform "
        "distribution.");
  }
}

void testInt0() {

  // WVP : probability greater than (1 - 2^(-40)).

  const auto a = Tensor::randomInt32(-1, 2, {10, 10}, 1011);
  const auto b = Tensor::randomInt32(-1, 2, {10, 10}, 1012);
  if (a.allEquivalent(b)) {
    throw poprithms::test::error(
        "a and b were created with different seeds, should be different");
  }
  if (!a.reduceMax({}).allEquivalent(Tensor::int32(1))) {
    throw poprithms::test::error(
        "100 values sampled uniformly from {-1, 0, +1}, should be "
        "one which is +1 with VHP");
  }

  if (!a.reduceMin({}).allEquivalent(Tensor::int32(-1))) {
    throw poprithms::test::error(
        "100 values sampled uniformly from {-1, 0, +1}, should be "
        "one which is -1 with VHP");
  }
}

void testBool() {
  const auto a =
      Tensor::randomBoolean({100}, 10100).toInt16().reduceSum({}).getInt16(0);

  if (a < 1 || a > 99) {
    throw poprithms::test::error(
        "100 coin flips, all came up heads? Unlikely. ");
  }
}

void testSampleWithoutReplacement() {

  uint64_t range{60};
  for (uint64_t n : {3, 30, 50}) {
    auto x = Tensor::sampleWithoutReplacementUnsigned64(range, n, 1011);
    x.assertType(DType::Unsigned64);
    auto vals  = x.getUnsigned8Vector();
    auto asSet = std::set<uint64_t>(vals.cbegin(), vals.cend());
    if (asSet.size() != vals.size()) {
      throw poprithms::test::error(
          "duplicates in testSampleWithoutReplacement");
    }
  }

  if (Tensor::sampleWithoutReplacementUnsigned64(10, 5, 1011)
          .allEquivalent(
              Tensor::sampleWithoutReplacementUnsigned64(10, 5, 1012))) {
    throw poprithms::test::error("Distinct seeds should result in distinct "
                                 "Tensors, in testSampleWithoutReplacement");
  }
}

void testMask() {

  auto x0 = Tensor::mask(DType::Int32, {10, 5, 2}, 6, 1011);
  auto x1 = Tensor::mask(DType::Int32, {10, 5, 2}, 6, 1012);

  x0.reduceSum().assertAllEquivalent(scalar(DType::Int32, 6));
  x1.reduceSum().assertAllEquivalent(scalar(DType::Int32, 6));

  const auto s = (x0 + x1).toBoolean().toInt64().reduceSum().getInt64(0);
  if (s < 7) {
    throw poprithms::test::error(
        "masks x0 and x1 should be distinct, their joint support "
        "should be larger than each of their individual supports, testMask ");
  }
}

void testSampleWithReplacement() {
  auto x0 = Tensor::sampleUnsigned64(Tensor::Replacement::Yes, {10}, 5, 1011)
                .getUnsigned64Vector();
  auto asSet = std::set<uint64_t>(x0.cbegin(), x0.cend());
  if (asSet.size() > 5) {
    throw poprithms::test::error(
        "Values were sampled from the range [0, 5), cannot be more than 5 "
        "distinct values, in testSampleWithReplacement");
  }
}

} // namespace

int main() {
  testFloat();
  testInt0();
  testBool();
  testSampleWithoutReplacement();
  testMask();
  testSampleWithReplacement();
}
