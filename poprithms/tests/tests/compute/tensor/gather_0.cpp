// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <iostream>
#include <iterator>
#include <vector>

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/memory/chain/settutil.hpp>
#include <poprithms/ndarray/strideandoffset.hpp>
#include <poprithms/ndarray/unfold.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms::compute::host;

using poprithms::compute::host::Tensor;

using U =
    poprithms::ndarray::Unfolder<Tensor,
                                 poprithms::ndarray::TUnfoldHelper<Tensor>>;

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

  // Baseline values from pytorch.

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
        .assertAllEquivalent(Tensor::int32({2, 2, 1}, {0, 1, 6, 7}));
  }

  {
    // 0 1       0 1     0 1
    // 2 3  ==>  2 3 ==> 2 3
    // 4 5       . .
    // 6 7       . .
    uint64_t dim{0};
    uint64_t size{2};
    for (uint64_t step : {3, 4, 5}) {
      auto x = U::unfold(t, dim, size, step);
      x.assertAllEquivalent(Tensor::int32({1, 2, 2}, {0, 2, 1, 3}));
    }

    U::unfold(t, dim, size, 2)
        .assertAllEquivalent(
            Tensor::int32({2, 2, 2}, {0, 2, 1, 3, 4, 6, 5, 7}));

    // 0 1
    // 2 3
    // 2 3
    // 4 5
    // 4 5
    // 6 7
    U::unfold(t, dim, size, 1)
        .assertAllEquivalent(
            Tensor::int32({3, 2, 2}, {0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7}));
  }
}

void testUnfold1() {
  // Example from
  // https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html
  auto t = Tensor::arangeInt32(1, 8, 1);

  U::unfold(t, 0, 2, 1)
      .assertAllEquivalent(
          Tensor::int32({6, 2}, {1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7}));

  U::unfold(t, 0, 2, 2)
      .assertAllEquivalent(Tensor::int32({3, 2}, {1, 2, 3, 4, 5, 6}));

  U::unfold(t, 0, /*size*/ 9, /*step*/ 1)
      .assertAllEquivalent(Tensor::int32({0, 9}, {}));

  // 1 2 3 4 5 6 7
  // =======
  //     =======
  U::unfold(t, 0, 4, 2)
      .assertAllEquivalent(Tensor::int32({2, 4}, {1, 2, 3, 4, 3, 4, 5, 6}));

  // 1 2 3 4 5 6 7
  // =======
  //       =======
  U::unfold(t, 0, 4, 3)
      .assertAllEquivalent(Tensor::int32({2, 4}, {1, 2, 3, 4, 4, 5, 6, 7}));

  // 1 2 3 4 5 6 7
  // =======
  //         =======
  U::unfold(t, 0, 4, 4)
      .assertAllEquivalent(Tensor::int32({1, 4}, {1, 2, 3, 4}));

  U::unfold(t, 0, 3, 4)
      .assertAllEquivalent(Tensor::int32({2, 3}, {1, 2, 3, 5, 6, 7}));
}

void testUnfold2() {

  auto t = Tensor::arangeInt32(0, 2 * 3 * 4, 1).reshape({2, 3, 4});

  U::unfold(t, 1, /*size*/ 1, /*step*/ 1)
      .assertAllEquivalent(t.reshape({2, 3, 4, 1}));

  U::unfold(t, 1, 0, 20).assertAllEquivalent(Tensor::int32({2, 1, 4, 0}, {}));
}

void testOffsetAndStrides0() {

  using HostHelper = poprithms::ndarray::TFromStrideAndOffsetHelper<Tensor>;

  auto s = poprithms::ndarray::FromStrideAndOffset<Tensor, HostHelper>();

  {

    auto t0 = Tensor::arangeInt32(0, 2 * 3 * 4, 1);

    // shape=(2,3,4),tData=(AllocData(dtype=int32,nelms=24),values=
    //
    // [[[ 0  1  2  3  ]
    //   [ 4  5  6  7  ]
    //   [ 8  9  10 11 ]]
    //  [[ 12 13 14 15 ]
    //   [ 16 17 18 19 ]
    //   [ 20 21 22 23 ]]]

    s.asStrided(t0, {12, 4, 1}, 0, {1, 2, 3})
        .assertAllEquivalent(
            t0.reshape({2, 3, 4}).slice({0, 0, 0}, {1, 2, 3}),
            "No reverse, no strides, offset = 0.");

    s.asStrided(t0, {12, 4, 1}, 12 + 4 + 1, {1, 2, 3})
        .assertAllEquivalent(
            t0.reshape({2, 3, 4}).slice({1, 1, 1}, {2, 3, 4}),
            "No reverse, no strides, offset != 0 (\"top-right\"corner.");

    s.asStrided(t0, {12, 4, 1}, 12 + 0 + 1, {1, 2, 3})
        .assertAllEquivalent(
            t0.reshape({2, 3, 4}).slice({1, 0, 1}, {2, 2, 4}),
            "No reverse, no strides, offset != 0 (1,0,1 offset by "
            "dimension).");

    s.asStrided(t0, {12, 4, 1}, 12 + 4 + 1, {1, 1, 1})
        .assertAllEquivalent(
            t0.reshape({2, 3, 4}).slice({1, 1, 1}, {2, 2, 2}),
            "Single (center) element slice");

    s.asStrided(t0, {12, 4, 1}, 2 * 3 * 4 - 1, {1, 1, 1})
        .assertAllEquivalent(Tensor::int32(2 * 3 * 4 - 1),
                             "Single (final) element slice");

    // [[[ 0  3  ]
    //   [ 8  11 ]]
    //  [[ 12 15 ]
    //   [ 20 23 ]]]
    s.asStrided(t0, {12, 8, 3}, 0, {2, 2, 2})
        .assertAllEquivalent(
            t0.reshape({2, 3, 4}).subSample({1, 2, 3}),
            "All the corners (has striding but not reverse) (1)");

    std::vector<int32_t> corners{0, 3, 8, 11, 12, 15, 20, 23};
    s.asStrided(t0, {12, 8, 3}, 0, {2, 2, 2})
        .assertAllEquivalent(
            Tensor::int32({2, 2, 2}, corners),
            "All the corners (has striding but not reverse) (2)");

    auto revCorners = corners;
    std::reverse(revCorners.begin(), revCorners.end());
    s.asStrided(t0, {-12, -8, -3}, 23, {2, 2, 2})
        .assertAllEquivalent(Tensor::int32({2, 2, 2}, revCorners),
                             "All the corners, reversed (no dimshuffle)");

    // a.dimShuffle(perm) has strides which are perm.applyTo(a.strides)
    //  ( 1 2 0 ) applied to {-12, -8, -3} is {-8, -3, -12}.
    s.asStrided(t0, {-8, -3, -12}, 23, {2, 2, 2})
        .assertAllEquivalent(
            Tensor::int32({2, 2, 2}, revCorners).dimShuffle({{1, 2, 0}}),
            "All the corners, reversed, dimshuffled");
  }
}

void testOffsetAndStrides1() {

  using HostHelper = poprithms::ndarray::TFromStrideAndOffsetHelper<Tensor>;

  auto s = poprithms::ndarray::FromStrideAndOffset<Tensor, HostHelper>();
  {

    // [[ 0  1  2  3  4  5  6  ]
    //  [ 7  8  9  10 11 12 13 ]
    //  [ 14 15 16 17 18 19 20 ]
    //  [ 21 22 23 24 25 26 27 ]
    //  [ 28 29 30 31 32 33 34 ]
    //  [ 35 36 37 38 39 40 41 ]]
    auto t0 = Tensor::arangeInt32(0, 42, 1).reshape({6, 7});

    // [[ 27 41 ]
    //  [ 24 38 ]]
    auto t1 = t0.slice({3, 3}, {6, 7})
                  .subSample({2, 3})
                  .reverse(1)
                  .dimShuffle({{1, 0}});

    t1.assertAllEquivalent(
        // The expectation is obtained by observing the output (above).
        s.asStrided(t0.flatten(), {-3, 14}, 27, {2, 2}),
        "mixed bag of transforms #1");
  }

  {

    auto t0 = Tensor::arangeInt32(0, 2 * 9 * 3, 1).reshape({2, 9, 3});

    // [[ 44 35 ]
    //  [ 43 34 ]
    //  [ 42 33 ]]
    //
    // with a bunch of arbitrary singleton dimensions.
    auto t1 = t0.slice({1, 2, 0}, {2, 8, 3})
                  .subSample({1, 3, 1})
                  .dimShuffle({{2, 0, 1}})
                  .reverse({0, 2})
                  .reshape({1, 3, 1, 2, 1});

    t1.assertAllEquivalent(
        s.asStrided(
            t0.flatten(), {100, -1, -100, -9, 100}, 44, {1, 3, 1, 2, 1}),
        "Test with arbitrary singleton dimensions (are they ignored?)");
  }

  {

    auto t0     = Tensor::int32({2, 2}, {1, 2, 3, 4});
    auto empty0 = t0.slice({0, 0}, {0, 2});
    empty0.reshape({5, 2, 0}).assertAllEquivalent(
        s.asStrided(empty0, {1, 2, 5}, 12, {5, 2, 0}), "empty slice test");
  }

  {
    auto t0 = Tensor::int32({1}, {7});
    s.asStrided(t0, {1, 2}, 0, {1, 1})
        .assertAllEquivalent(Tensor::int32({1, 1}, {7}));
  }
}

// For the next test, we do some mocking to track the number of concats used
// in the lowering from ptr-strides-offset to a poplar-like API.
class ConcatTracker
    : public poprithms::ndarray::TFromStrideAndOffsetHelper<Tensor> {
public:
  static Tensor concat(const std::vector<Tensor> &ts, uint64_t d) {
    if (ts.size() == 1) {
      return ts[0];
    }
    ++nConcats;
    return Tensor::concat(ts, d);
  }
  static int nConcats;
};

int ConcatTracker::nConcats = 0;

void testOffsetAndStrides2() {

  using HostHelper = ConcatTracker;

  auto s = poprithms::ndarray::FromStrideAndOffset<Tensor, HostHelper>();
  {
    // 0  1  2 .... 9
    // 10 11 12 ... 19
    auto t0 = Tensor::arangeInt32(0, 20, 1).reshape({2, 10});

    // sampling with stride of 2, go from shape [2,10] to shape [2,5].
    // As 10 is divisible by 2, no concats are required.

    // 0  2  ... 8
    // 10 12 ... 18
    auto t1 = s.asStrided(t0.flatten(), {10, 2}, 0, {2, 5});
    if (ConcatTracker::nConcats != 0) {
      throw poprithms::test::error(
          "After the sub-sample with a stride which divides the dimension, "
          "no concats should have been required");
    }

    // 0  3  ... 9
    // 10 13 ... 19
    t1 = s.asStrided(t0.flatten(), {10, 3}, 0, {2, 4});
    if (ConcatTracker::nConcats != 1) {
      throw poprithms::test::error(
          "After the sub-sample with a stride which doen not divides the "
          "dimension, 1 concat should have been required");
    }
  }
}

} // namespace

int main() {
  testGather0();
  testGather1();
  testUnfold0();
  testUnfold1();
  testUnfold2();
  testOffsetAndStrides0();
  testOffsetAndStrides1();
  testOffsetAndStrides2();
  return 0;
}
