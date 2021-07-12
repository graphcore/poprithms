// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <cmath>

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>

namespace {
using namespace poprithms::compute::host;

void testMod0() {
  const auto a = Tensor::float16(5.5);
  const auto b = Tensor::float16(2.0);
  const auto c = a % b;
  c.assertAllEquivalent(Tensor::float16(1.5));
}

void testSign0() {
  Tensor::float32({4}, {-0.5, 0.0, 0.4, -12.})
      .sign()
      .assertAllEquivalent(Tensor::float32({4}, {-1, 0, 1, -1}));

  Tensor::float32({4}, {0.01, -100., 0.1, 0.000})
      .sign()
      .assertAllEquivalent(Tensor::float32({4}, {+1., -1, +1, 0.}));
}

void testSign1() {
  Tensor::unsigned32({3}, {100, 0, 1})
      .sign()
      .assertAllEquivalent(Tensor::unsigned32({3}, {1, 0, 1}));

  Tensor::boolean({3}, {false, false, true})
      .sign()
      .assertAllEquivalent(Tensor::boolean({3}, {false, false, true}));
}

constexpr double pi = 3.14159265358979323846;
void testSin0() {
  auto x        = Tensor::float64({4}, {0., pi / 6, pi / 4, 3 * pi / 2.});
  auto expected = Tensor::float64({4}, {0., 1 / 2., 1. / std::sqrt(2.), -1.});
  x.sin().assertAllClose(expected, 0.001, 0.001);
  x.sin_();
  x.assertAllClose(expected, 0.001, 0.001);
}

void testCos0() {

  auto a = Tensor::float64({2}, {0., pi / 6});
  auto b = Tensor::float64({2}, {pi / 4, 3 * pi / 2.});
  auto x = Tensor::concat_({a, b}, 0);
  auto expected =
      Tensor::float64({4}, {1., std::sqrt(3.) / 2, 1. / std::sqrt(2), 0.});
  x.cos().assertAllClose(expected, 0.001, 0.001);
  x.cos_();
  x.assertAllClose(expected, 0.001, 0.001);
}

} // namespace

int main() {
  testMod0();
  testSign0();
  testSign1();
  testSin0();
  testCos0();
  return 0;
}
