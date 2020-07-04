// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/sett.hpp>

namespace {
using namespace poprithms::memory::nest;

void test0() {
  auto base = poprithms::memory::nest::Sett{{{{2, 2, 0}}}};
  //        xx..xx..
  //        ------->
  //  <-------
  auto rev = base.getReverse(2);
  base.confirmEquivalent(rev);
}

void test1() {
  //  xx.xx....x.xxx.xx....x.xxx.xx....
  //  -------------------------------->
  //  <-----------
  //  reflection looks like this:
  //  x.x...xx.xxx.x...xx.xxx.x...
  //
  poprithms::memory::nest::Sett base{{{{8, 4, -2}, {3, 1, 2}}}};
  poprithms::memory::nest::Sett expected{{{{8, 4, 6}, {3, 1, -1}}}};
  auto rev = base.getReverse(12);
  rev.confirmEquivalent(expected);
}

void test2() {
  poprithms::memory::nest::Sett base{{}};
  auto rev = base.getReverse(133);
  if (rev.hasStripes()) {
    throw error("reverse of no-stripes is no-stripes");
  }
}

void test3() {
  poprithms::memory::nest::Sett base{
      {{{1000, 1000, 1}, {100, 100, 1}, {10, 10, 1}, {1, 1, 1}}}};
  auto rev = base.getReverse(1);
  poprithms::memory::nest::Sett expected{
      {{{1000, 1000, 1000}, {100, 100, 99}, {10, 10, 9}, {1, 1, 0}}}};
  expected.confirmEquivalent(rev);
}

void test4() {
  // xx.xx...xx.xx...xx.xx...xx.xx...
  poprithms::memory::nest::Sett base{{{{5, 3, 0}, {2, 1, 0}}}};
  auto rev = base.getReverse(1);
  poprithms::memory::nest::Sett expected{{{{5, 3, 4}, {2, 1, 0}}}};
  expected.confirmEquivalent(rev);
}
} // namespace

int main() {
  test0();
  test1();
  test2();
  test3();
  test4();
  return 0;
}
