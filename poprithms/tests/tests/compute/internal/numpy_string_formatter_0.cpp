// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <random>

#include <compute/host/include/numpyformatter.hpp>
#include <poprithms/ndarray/dtype.hpp>

namespace {

using namespace poprithms::compute::host;
using namespace poprithms::ndarray;

// generate a vector of random "numbers"
std::vector<std::string> getFrags(uint64_t n) {
  static std::mt19937 gen(1011);
  std::vector<std::string> frags(n);
  std::array<char, 10> chars{
      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
  for (uint64_t i = 0; i < n; ++i) {

    // fragments are one of
    // {0, 01, 012, 0123}
    const auto L = 1 + gen() % 4;
    frags[i]     = std::string(chars.cbegin(), std::next(chars.cbegin(), L));
  }
  return frags;
}

void print(const Shape &s, uint64_t t) {
  std::ostringstream oss;
  NumpyFormatter::append(getFrags(s.nelms_u64()), oss, s, t);
  std::cout << "@shape=" << s << ":\n" << oss.str() << std::endl;
}
} // namespace

int main() {

  //  @shape=(7):
  //  [ 0123
  //    012
  //    0
  //    01
  //    01
  //    0123
  //    012  ]
  print({7}, 100);

  //   @shape=(2,3):
  //   [[ 0    0123 0  ]
  //    [ 0123 0123 01 ]]
  print({2, 3}, 100);

  //   @shape=(3,2):
  //   [[ 01 0123 ]
  //    [ 0  0123 ]
  //    [ 0  01   ]]
  print({3, 2}, 100);

  //
  //   @shape=(3,2,1,1):
  //   [[[[ 012 ]]
  //     [[ 01  ]]]
  //    [[[ 01  ]]
  //     [[ 01  ]]]
  //    [[[ 012 ]]
  //     [[ 01  ]]]]
  print({3, 2, 1, 1}, 100);

  //
  //   @shape=(2,2,2):
  //   [[[ 0123 01 ]
  //     [ 0    01 ]]
  //    [[ 0123 01 ]
  //     [ 0    01 ]]]
  print({2, 2, 2}, 100);

  //
  //   @shape=():
  //   scalar(0)
  print({}, 100);

  //
  //   @shape=(200,200,200):
  //   (01,...(7999998 more values)...,0)
  print({200, 200, 200}, 5);

  //
  //   @shape=(200,200,200):
  //   (012,01,...(7999996 more values)...,01,012)
  print({200, 200, 200}, 6);
}
