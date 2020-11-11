// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <numeric>

#include <poprithms/compute/host/error.hpp>
#include <poprithms/compute/host/tensor.hpp>

namespace {
using namespace poprithms::compute::host;
void dimShuffleTest0() {
  const auto t0 = Tensor::arangeInt16(0, 2 * 3 * 5, 1);
  const auto r0 = t0.reshape({2, 3, 5});
  const auto d0 = r0.dimShuffle({{1, 2, 0}});
  if (d0.shape() != Shape{3, 5, 2}) {
    throw error("Incorrect Shape after dimShuffle (in test)");
  }
  std::vector<int16_t> expected{0,  15, 1,  16, 2,  17, 3,  18, 4,  19,
                                5,  20, 6,  21, 7,  22, 8,  23, 9,  24,
                                10, 25, 11, 26, 12, 27, 13, 28, 14, 29};
  d0.assertAllEquivalent(Tensor::int16({3, 5, 2}, expected.data()));
}

void dimShuffleTest1() {
  auto t0_ = Tensor::uniformFloat64(-10., 10., {3, 4, 5, 6}, 1011).toInt32();
  auto t0  = t0_.copy();
  for (uint64_t i = 0; i < 4; ++i) {
    t0 = t0.dimShuffle_({{1, 2, 3, 0}});
    t0 = t0.add_(Tensor::int32(1));
  }
  auto diff = t0 - t0_ - Tensor::int32(4);
  diff.assertAllEquivalent(Tensor::int32(0));
}

void gatherTest0() {
  const auto t0 = Tensor::arangeInt16(0, 2 * 3 * 5, 1);
  const auto r0 = t0.reshape({2, 3, 5});
  // 20 22 22
  const auto g0 = r0.gather(1, {1}).gather(2, {0, 2, 2}).gather(0, {1});
  g0.assertAllEquivalent(Tensor::int16({3}, {20, 22, 22}));
}

} // namespace

int main() {
  dimShuffleTest0();
  dimShuffleTest1();
  gatherTest0();
  return 0;
}
