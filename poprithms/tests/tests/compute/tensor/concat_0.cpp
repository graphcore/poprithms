// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <numeric>

#include <poprithms/compute/host/error.hpp>
#include <poprithms/compute/host/tensor.hpp>

namespace {
using namespace poprithms::compute::host;

void test0() {
  const auto t0 = Tensor::arangeInt8(0, 4, 1).reshape_({4, 1});
  std::cout << "a" << std::endl;
  const auto t1 = Tensor::arangeInt8(10, 14, 1).reshape_({4, 1});

  std::cout << "b" << std::endl;
  const auto t2 = Tensor::arangeInt8(20, 24, 1).reshape_({4, 1});

  std::cout << "c" << std::endl;
  const auto t3 = concat({t0, t1, t2}, 1);

  std::cout << "d" << std::endl;
  const auto t4 = Tensor::arangeInt8(30, 45, 1).reshape({5, 3});

  std::cout << "e" << std::endl;
  const auto t5 = concat_({t3, t4}, 0);

  std::cout << "f" << std::endl;
  if (t5.shape() != Shape{9, 3}) {
    throw error("Incorrect shape after concat in test0");
  }

  std::cout << "g" << std::endl;

  std::vector<int> expected{0,  10, 20, //
                            1,  11, 21, //
                            2,  12, 22, //
                            3,  13, 23, //

                            30, 31, 32, //
                            33, 34, 35, //
                            36, 37, 38, //
                            39, 40, 41, //
                            42, 43, 44};

  t5.assertAllEquivalent(Tensor::refInt32({9, 3}, expected.data()));

  std::cout << "h" << std::endl;
}
} // namespace

int main() {
  test0();
  return 0;
}
