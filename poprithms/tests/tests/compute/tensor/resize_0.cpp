// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>

namespace {
using namespace poprithms::compute::host;

void testDimRoll0() {
  const auto a      = Tensor::arangeInt32(0, 2 * 3 * 5, 1).reshape({2, 3, 5});
  const auto rolled = a.dimRoll(Dimension(0), Dimension(2));
  const auto permed = a.dimShuffle({{1, 2, 0}});
  permed.assertAllEquivalent(rolled);
}

void testDimRoll1() {
  const auto a      = Tensor::arangeInt32(0, 2 * 3 * 5, 1).reshape({2, 3, 5});
  const auto rolled = a.dimRoll(Dimension(2), Dimension(0));
  const auto permed = a.dimShuffle({{2, 0, 1}});
  permed.assertAllEquivalent(rolled);
}

void testDimRoll2() {
  const auto a      = Tensor::arangeInt32(0, 2 * 3 * 5, 1).reshape({2, 3, 5});
  const auto rolled = a.dimRoll(Dimension(1), Dimension(0));
  const auto permed = a.dimShuffle({{1, 0, 2}});
  permed.assertAllEquivalent(rolled);
}

void testResize0() {

  //  shape=(2,3),tData=(AllocData(dtype=int32,nelms=6),values=
  //  [[ 0 1 2 ]
  //   [ 3 4 5 ]]
  auto a = Tensor::arangeInt32(0, 2 * 3, 1).reshape({2, 3});

  //  shape=(4,6),tData=(AllocData(dtype=int32,nelms=24),values=
  //  [[ 0 0 1 1 2 2 ]
  //   [ 0 0 1 1 2 2 ]
  //   [ 3 3 4 4 5 5 ]
  //   [ 3 3 4 4 5 5 ]]
  auto b = a.resize(Dimension(0), Stride(2)).resize(Dimension(1), Stride(2));

  b.assertAllEquivalent(
      Tensor::int32({4, 6}, {0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2,
                             3, 3, 4, 4, 5, 5, 3, 3, 4, 4, 5, 5}));
}

void testResize1() {

  //  shape=(2,3,2),tData=(AllocData(dtype=int32,nelms=12),values=
  //  [[[ 0  1  ]
  //    [ 2  3  ]
  //    [ 4  5  ]]
  //   [[ 6  7  ]
  //    [ 8  9  ]
  //    [ 10 11 ]]]
  auto a = Tensor::arangeInt32(0, 2 * 3 * 2, 1).reshape({2, 3, 2});

  //  shape=(2,6,4),tData=(AllocData(dtype=int32,nelms=48),values=
  //  [[[ 0  0  1  1  ]
  //    [ 0  0  1  1  ]
  //    [ 2  2  3  3  ]
  //    [ 2  2  3  3  ]
  //    [ 4  4  5  5  ]
  //    [ 4  4  5  5  ]]
  //   [[ 6  6  7  7  ]
  //    [ 6  6  7  7  ]
  //    [ 8  8  9  9  ]
  //    [ 8  8  9  9  ]
  //    [ 10 10 11 11 ]
  //    [ 10 10 11 11 ]]]
  auto b = a.resize(Dimension(1), Stride(2)).resize(Dimension(2), Stride(2));

  b.assertAllEquivalent(Tensor::int32({2, 6, 4},
                                      {
                                          0,  0,  1,  1,  0,  0,  1,  1, //
                                          2,  2,  3,  3,  2,  2,  3,  3, //
                                          4,  4,  5,  5,  4,  4,  5,  5, //
                                          6,  6,  7,  7,  6,  6,  7,  7, //
                                          8,  8,  9,  9,  8,  8,  9,  9, //
                                          10, 10, 11, 11, 10, 10, 11, 11 //
                                      }));
}

void testResize2() {
  const auto a  = Tensor::arangeInt32(0, 2 * 3 * 5, 1).reshape({2, 3, 5});
  const auto a0 = a.copy();
  const auto b  = a.resize_(Dimension(0), Stride(5))
                     .resize_(Dimension(1), Stride(3))
                     .resize_(Dimension(2), Stride(2))
                     .subSample_({5, 3, 2})
                     .mul_(Tensor::int32(2));
  a.assertAllEquivalent(a0.mul(Tensor::int32(2)));
}

} // namespace

int main() {
  testResize0();
  testResize1();
  testResize2();
  testDimRoll0();
  testDimRoll1();
  testDimRoll2();
  return 0;
}
