// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/nest/sett.hpp>

namespace {

void test0() {

  using namespace poprithms::memory::nest;

  //
  // 11111.....1111111111.....1111111111.....1111111111.....11111111 (10, 5,
  // 10)
  //           1.111111.1                                            (6,1,2)
  // 1.1.1     1 1.11.1 1     1.1.11.1.1     1.1.11.1.1     1.1.11.1 (2,1,2)
  // 012345678901234567890123456789012345678901234567890123456789012
  // 0         1         2         3         4         5         6
  //
  // 0 1 2     3 4 56 7 8     9 0 12 3 4     5 6 78 9 0     1 2 34 5
  //                            1                     2
  //

  Sett sett({{{{10, 5, 10}, {6, 1, 2}, {2, 1, 2}}}});

  // The indices where there are ons (hand calculated from diagram above).
  std::vector<int64_t> cuts{0,  2,  4,  10, 12, 14, 15, 17, 19,
                            25, 27, 29, 30, 32, 34, 40, 42, 44,
                            45, 47, 49, 55, 57, 59, 60, 62};

  for (uint64_t i0 = 0; i0 < cuts.size() - 1; ++i0) {
    int64_t x0 = cuts[i0] + 1;
    int64_t x1 = cuts[i0 + 1];
    for (auto x = x0; x <= x1; ++x) {
      if (sett.find(x) != x1) {
        std::ostringstream oss;
        oss << "Failure in test of Sett:find. ";
        oss << "Expected " << sett << ".find(" << x << ") to be " << x1
            << ", not " << sett.find(x);
        throw poprithms::test::error(oss.str());
      }
    }
  }
}

void test1() {

  using namespace poprithms::memory::nest;

  //  432101234
  //..1.1..1.1..1.1
  Sett sett({{{{3, 2, 1}, {1, 1, 0}}}});

  std::vector<std::array<int64_t, 2>> expect{
      {-4, -4}, {-3, -2}, {-2, -2}, {-1, 1}, {0, 1}, {1, 1}};

  for (auto &[i, f] : expect) {
    const auto found = sett.find(f);
    if (found != f) {
      std::ostringstream oss;
      oss << "Failure in negative case find. "
          << "Expected sett.find(" << i << ") = " << f << ", not " << found;
      throw poprithms::test::error(oss.str());
    }
  }
}

} // namespace

int main() {
  test0();
  test1();
}
