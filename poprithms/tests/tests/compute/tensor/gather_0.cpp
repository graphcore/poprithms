// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/ndarray/unfold.hpp>

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

void testUnfold0() {

  using poprithms::compute::host::Tensor;

  using U =
      poprithms::ndarray::Unfolder<Tensor,
                                   poprithms::ndarray::TUnfoldHelper<Tensor>>;

  // 0 1
  // 2 3
  // 4 5
  // 6 7
  auto t = Tensor::arangeInt32(0, 8, 1).reshape({4, 2});

  {
    // 0 1       0 1     0 1
    // 2 3  ==>  . . ==> 6 7
    // 4 5       . .
    // 6 7       6 7
    uint64_t dim{0};
    uint64_t size{1};
    uint64_t step{3};
    U::unfold(t, dim, size, step)
        .assertAllEquivalent(Tensor::int32({2, 2}, {0, 1, 6, 7}));
  }

  {
    // 0 1       0 1     0 1
    // 2 3  ==>  2 3 ==> 2 3
    // 4 5       . .
    // 6 7       . .
    uint64_t dim{0};
    uint64_t size{2};
    for (uint64_t step : {4, 5}) {
      auto x = U::unfold(t, dim, size, step);
      x.assertAllEquivalent(Tensor::int32({2, 2}, {0, 1, 2, 3}));
    }

    U::unfold(t, dim, size, 3)
        .assertAllEquivalent(Tensor::int32({3, 2}, {0, 1, 2, 3, 6, 7}));

    U::unfold(t, dim, size, 2)
        .assertAllEquivalent(Tensor::int32({4, 2}, {0, 1, 2, 3, 4, 5, 6, 7}));

    // 0 1
    // 2 3
    // 2 3
    // 4 5
    // 4 5
    // 6 7
    U::unfold(t, dim, size, 1)
        .assertAllEquivalent(
            Tensor::int32({6, 2}, {0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7}));
  }
}

} // namespace

int main() {
  testGather0();
  testGather1();
  testUnfold0();
  return 0;
}
