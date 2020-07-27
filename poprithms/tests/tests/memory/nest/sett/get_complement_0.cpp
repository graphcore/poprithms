// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/sett.hpp>

namespace {
using namespace poprithms::memory::nest;

void confirm(const Sett &s, const DisjointSetts &ec) {
  const auto c = s.getComplement();
  std::cout << c << std::endl;
  if (!c.equivalent(ec)) {
    std::ostringstream oss;
    oss << "Failed to compute the complement of " << s << " correctly. "
        << "Expected " << ec << ", but observed " << c << '.';
    throw error(oss.str());
  }
}

void test0() { confirm({{{2, 3, 0}}}, {{{{3, 2, 2}}}}); }

void test1() {

  // 11...1111111...1111111... (7,3,5)
  // 1....1...11....1...11.... (2,3,4)
  //
  // .1111.111..1111.111..1111
  confirm({{{7, 3, 5}, {2, 3, 4}}},
          DisjointSetts{{Sett{{{4, 6, 1}}}, Sett{{{3, 7, 6}}}}});
}

void test2() {

  // 1.1111111111.1111111111.1111111111. (10, 1, 2)
  // 1.1111111111 1111111111 1111111111  (6, 0, 3)
  // 1..11111.111..11111.111..11111.111  (3, 1, 0)
  //
  // .11.....1...11.....1...11.....1...
  //
  confirm({{{10, 1, 2}, {6, 0, 3}, {3, 1, 0}}},
          DisjointSetts{{Sett{{{2, 9, 1}}}, Sett{{{1, 10, 8}}}}});
}

void test3() {
  // ..11111.....
  // ....11111...
  const Sett a({{{{5, 5, 2}}}});
  const Sett b({{{{5, 5, 4}}}});
  const auto diff = a.subtract(b);
  if (!Sett({{{{2, 8, 2}}}}).equivalent(diff)) {
    std::cout << diff << std::endl;
    throw error("Failed in basic test of subtract");
  }
}

} // namespace

int main() {
  test0();
  test1();
  test2();
  test3();
  return 0;
}
