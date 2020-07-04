// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/sett.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms::memory::nest;

void test0() {

  Sett sett0{{{1250, 0, 0}, {1112, 70, 771}, {98, 261, 58}, {14, 67, 70}}};
  Sett sett1{{{1250, 0, 0}, {168, 954, 52}, {89, 10, 44}, {17, 11, 2}}};
  auto sampled0 = sett0.sampleAt(sett1);
}

void test1() {
  Sett sett0{{{23375, 0, 0}, {1431, 240, 675}, {40, 73, 88}, {10, 15, 7}}};
  Sett sett1{{{23375, 0, 0}, {1802, 82, 646}, {284, 86, 302}, {178, 64, 39}}};
  auto sampled0 = sett0.sampleAt(sett1);
}

} // namespace

int main() {
  test0();
  test1();
  return 0;
}
