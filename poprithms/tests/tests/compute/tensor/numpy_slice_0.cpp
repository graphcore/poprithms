// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <numeric>

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>

namespace {
using namespace poprithms::compute::host;

void test0() {

  // In [27]: X = np.arange(8).reshape(2,4)
  //
  // In [28]: X
  // Out[28]:
  // array([[0, 1, 2, 3],
  //        [4, 5, 6, 7]])
  const auto t = Tensor::int32({2, 4}, {0, 1, 2, 3, 4, 5, 6, 7});

  // In [31]: print(X[0:2:1, 0:3:1])
  // [[0 1 2]
  //  [4 5 6]]
  t.slice(Starts({0, 0}), Ends({2, 3}), {}, {})
      .assertAllEquivalent(Tensor::int32({2, 3}, {0, 1, 2, 4, 5, 6}));

  // In [33]: print(X[:, 0:4:2])
  // [[0 2]
  //  [4 6]]
  t.slice(Starts({0}), Ends({4}), Steps({2}), Dims({1}))
      .assertAllEquivalent(Tensor::int32({2, 2}, {0, 2, 4, 6}));

  // In [34]: print(X[1:2:1, 3:-10:-1])
  // [[7 6 5 4]]
  t.slice(Starts({3, 1}), Ends({-10, 2}), Steps({-1, 1}), Dims({1, 0}))
      .assertAllEquivalent(Tensor::int32({1, 4}, {7, 6, 5, 4}));
}

void test1() {

  // [ 5 7 ]
  const auto t = Tensor::int64({2}, {5, 7});

  //  t[1:-1:-1]
  t.slice(Starts({1}), Ends({-1}), Steps({-1}), {})
      .assertAllEquivalent(Tensor::int64({0}, {}));

  //  t[1:-2:-1]
  t.slice(Starts({1}), Ends({-2}), Steps({-1}), {})
      .assertAllEquivalent(Tensor::int64({1}, {7}));

  //  t[1:-3:-1]
  t.slice(Starts({1}), Ends({-3}), Steps({-1}), {})
      .assertAllEquivalent(Tensor::int64({2}, {7, 5}));

  //  t[1:0:+1]
  t.slice(Starts({1}), Ends({0}), Steps({+1}), {})
      .assertAllEquivalent(Tensor::int64({0}, {}));

  //  t[1:1:1]
  t.slice(Starts({1}), Ends({1}), Steps({+1}), {})
      .assertAllEquivalent(Tensor::int64({0}, {}));

  //  t[1:2:1]
  t.slice(Starts({1}), Ends({2}), Steps({+1}), {})
      .assertAllEquivalent(Tensor::int64({1}, {7}));

  //  t[-100:100:1]
  t.slice(Starts({-100}), Ends({+100}), Steps({+1}), {})
      .assertAllEquivalent(Tensor::int64({2}, {5, 7}));
}

void test2() {
  const auto t = Tensor::arangeInt32(0, 2 * 3 * 5, 1).reshape({2, 3, 5});
  const auto x0 =
      t.slice(Starts({0, 1}), Ends({4, 3}), Steps({1, 1}), Dims({2, 1}));
  const auto x1 = t.slice({0, 1, 0}, {2, 3, 4});
  x0.assertAllEquivalent(x1);
}

void test3() {
  const auto t = Tensor::arangeUnsigned32(0, 30, 1).reshape({5, 6});
  const auto x0 =
      t.slice(Starts({4, 5}), Ends({-100, -100}), Steps({-2, -2}), {});
  const auto x1 = t.reverse_({0, 1}).subSample({2, 2});
  x0.assertAllEquivalent(x1);
}

} // namespace

int main() {
  test0();
  test1();
  test2();
  test3();
  return 0;
}
