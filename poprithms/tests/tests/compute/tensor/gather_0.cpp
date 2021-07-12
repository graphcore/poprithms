// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>

namespace {
using namespace poprithms::compute::host;

void testGather0() {
  //
  //   +-----+
  //   | 0 1 |
  //   +-----+
  //     2 3
  //   +-----+
  //   | 4 5 |
  //   +-----+
  //     6 7
  //   +-----+
  //   | 8 9 |
  //   +-----+
  //
  auto x = Tensor::arangeInt32(0, 10, 1).reshape({5, 2}).gather(0, {0, 2, 4});
  x.assertAllEquivalent(Tensor::int32({3, 2}, {0, 1, 4, 5, 8, 9}));
}

void testGather1() {
  auto x = Tensor::arangeInt32(0, 10, 1).reshape_({5, 2});
  auto y = x.gather_(0, {0, 2, 4}).mul_(Tensor::int32(0));
  x.assertAllEquivalent(
      Tensor::int32({5, 2}, {0, 0, 2, 3, 0, 0, 6, 7, 0, 0}));
}

} // namespace

int main() {
  testGather0();
  testGather1();
  return 0;
}
