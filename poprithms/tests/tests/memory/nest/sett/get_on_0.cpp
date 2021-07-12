// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/error/error.hpp>
#include <poprithms/memory/nest/sett.hpp>

namespace {

using namespace poprithms::memory::nest;

void assertOn(const Sett &p, uint64_t i, int64_t j) {

  const auto computed = p.getOn(i);
  if (computed != j) {
    std::ostringstream oss;
    oss << "Expected " << p << ".getOn(" << i << ") to be " << j << ", not "
        << computed << '.';
    throw poprithms::test::error(oss.str());
  }
}
} // namespace

int main() {

  using namespace poprithms::memory::nest;

  assertOn({{{1, 493, 369}}}, 0, 369);

  Sett p1{{{{5, 5, 0}}}};
  // xxxxx.....xxxxx.....xxxxx
  // 01234     56789

  assertOn(p1, 0, 0);
  assertOn(p1, 1, 1);
  assertOn(p1, 4, 4);
  assertOn(p1, 5, 10);
  assertOn(p1, 26, 51);

  Sett p2{{{{7, 5, 0}, {2, 1, 1}}}};
  // xxxxxxx.....xxxxxxx.....xxxxxxx.....xxxxxxx.....
  //.xx.xx.......xx.xx.......xx.xx.......xx.xx......
  // 01 23       45 67       89

  assertOn(p2, 0, 1);
  assertOn(p2, 1, 2);
  assertOn(p2, 2, 4);
  assertOn(p2, 3, 5);
  assertOn(p2, 6, 7 + 5 + 1 + 2 + 1);

  // xx...xxxxxxx...xxxxxxx...xxxxxxx...xxxxxxx... (7, 3, 5)
  //       xxx.xx    xxx.xx    xxx.xx    xxx.xx    (3, 1, 1)
  // x.    x.x x.    x.x x.    x.x x.    x.x x.    (1, 1, 0)
  // 0     1 2 3     4 5 6     7 8 9
  // 0123456789                          0 1 2
  //           0123456789
  //                     0123456789
  //                               0123456789
  //                                         0123456780
  //

  Sett p3{{{{7, 3, 5}, {3, 1, 1}, {1, 1, 0}}}};
  assertOn(p3, 0, 0);
  assertOn(p3, 1, 6);
  assertOn(p3, 2, 8);
  assertOn(p3, 3, 10);
  assertOn(p3, 4, 16);
  assertOn(p3, 5, 18);
  assertOn(p3, 6, 20);
  assertOn(p3, 7, 26);
  assertOn(p3, 8, 28);
  assertOn(p3, 9, 30);

  assertOn({{{1, 9, 5}}}, 0, 5);
  assertOn({{{1, 9, 5}}}, 1, 15);
  assertOn({{{1, 9, 5}}}, 2, 25);

  // Examples in sett.hpp
  // 1...1111...1111...1111
  assertOn({{{4, 3, 4}}}, 0, 0);
  assertOn({{{4, 3, 4}}}, 1, 4);

  //.11.11.11.11.11.11.11.
  assertOn({{{2, 1, 1}}}, -3, -4);
  assertOn({{{2, 1, 1}}}, -2, -2);
  assertOn({{{2, 1, 1}}}, -1, -1);
  assertOn({{{2, 1, 1}}}, 0, 1);
  assertOn({{{2, 1, 1}}}, 1, 2);
  assertOn({{{2, 1, 1}}}, 2, 4);

  assertOn({{{1, 99, 0}}}, -2, -200);
  assertOn({{{1, 99, 0}}}, -1, -100);
  assertOn({{{1, 99, 0}}}, 0, 0);
  assertOn({{{1, 99, 0}}}, 1, +100);
  assertOn({{{1, 99, 0}}}, 2, +200);

  assertOn({{{1, 99, 17}}}, -1, -100 + 17);
  assertOn({{{1, 99, 17}}}, 0, 17);
  assertOn({{{1, 99, 17}}}, 1, +100 + 17);

  assertOn({{{1, 99, -17}}}, -1, -17);
  assertOn({{{1, 99, -17}}}, 0, 100 - 17);
  assertOn({{{1, 99, -17}}}, 1, +200 - 17);

  // 9 87   6 54   3 21   0 12   3 45
  // 1.11...1.11...1.11...1.11...1.11
  //           9876543210123456789

  Sett foo{{{{4, 3, 2}, {2, 1, 2}}}};
  assertOn(foo, 0, 2);
  assertOn(foo, 1, 4);
  assertOn(foo, 2, 5);
  assertOn(foo, 3, 9);
  assertOn(foo, -1, -2);
  assertOn(foo, -2, -3);
  assertOn(foo, -3, -5);
  assertOn(foo, -4, -9);
  assertOn(foo, -5, -10);

  return 0;
}
