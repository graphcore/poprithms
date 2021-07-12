// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>

namespace {
using namespace poprithms::compute::host;

void testScatterToZero0() {

  //    [[ 0 1 ]
  //     [ 2 3 ]
  //     [ 4 5 ]]
  auto x = Tensor::arangeInt32(0, 6, 1).reshape({2, 3});

  //     [[ 0 . . 1 2 ]
  //      [ . . . . . ]
  //      [ 3 . . 4 5 ]]
  const auto scatty = x.scatterToZero({3, 5}, {{0, 2}, {0, 3, 4}});

  scatty.assertAllEquivalent(
      Tensor::int32({3, 5}, {0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 3, 0, 0, 4, 5}));
}

void testScatter0() {

  //    [[ 0 1 ]
  //     [ 2 3 ]
  //     [ 4 5 ]]
  auto x = Tensor::arangeFloat16(0, 6, 1).reshape({2, 3});

  //     [[ 7 7 7 7 7 ]
  //      [ 7 7 7 7 7 ]
  //      [ 7 7 7 7 7 ]]
  auto target = Tensor::float16(7).expand({3, 5});

  //     [[ 0 7 7 1 2 ]
  //      [ 7 7 7 7 7 ]
  //      [ 3 7 7 4 5 ]]
  const auto scatty = x.scatterTo(target, {{0, 2}, {0, 3, 4}}).toInt32();

  scatty.assertAllEquivalent(
      Tensor::int32({3, 5}, {0, 7, 7, 1, 2, 7, 7, 7, 7, 7, 3, 7, 7, 4, 5}));
}

// It is possible to reverse the order when scattering
void testScatter1() {
  auto c = Tensor::arangeInt32(1, 5, 1).reshape({2, 2});
  auto b = Tensor::int32(7).expand({2, 3});
  auto a = c.scatterTo(b, {{1, 0}, {0, 2}});
  a.assertAllEquivalent(Tensor::int32({2, 3}, {3, 7, 4, 1, 7, 2}));
}

} // namespace

int main() {
  testScatterToZero0();
  testScatter0();
  testScatter1();
  return 0;
}
