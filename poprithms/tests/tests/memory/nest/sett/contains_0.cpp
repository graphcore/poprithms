// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <numeric>
#include <sstream>

#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/sett.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms::memory::nest;

void testContains(bool expected, const Sett &a, const Sett &b) {
  auto computed = a.contains(b);
  if (computed != expected) {
    std::ostringstream oss;
    oss << "In testContains, testing that " << a << ".contains(" << b
        << ") = " << expected << ". But is does not. ";
    throw error(oss.str());
  }
}

} // namespace

void test0() {

  // always on contains everything
  testContains(1, {{}}, {{{1, 10, 5}}});
  testContains(1, {{}}, {{{1, 0, 0}}});
  testContains(1, {{}}, {{}});
  testContains(1, {{}}, {{{0, 5, 0}}});

  // always off is contained in nothing, except always off
  testContains(0, {{{0, 1, 0}}}, {{{1, 10, 5}}});
  testContains(0, {{{0, 1, 0}}}, {{{1, 0, 0}}});
  testContains(0, {{{0, 1, 0}}}, {{{1, 5, 0}}});
  testContains(1, {{{0, 1, 0}}}, {{{0, 1, 0}}});
  testContains(1, {{{0, 1, 0}}}, {{{0, 5, 3}, {0, 2, 1}, {1, 0, 0}}});

  // co-prime mixed: never complete containment
  testContains(0,
               {{{145, 3, 45}, {55, 2, 101}}},
               {{{145, 4, 99}, {5, 5, 2}, {1, 1, 1}}});

  // one more simple example period 148 containing period 74
  testContains(1, {{{145, 3, 45}, {3, 1, 0}}}, {{{71, 3, 45}, {1, 1, 0}}});
  // reverse of the above set
  testContains(0, {{{71, 3, 45}, {1, 1, 0}}}, {{{145, 3, 45}, {3, 1, 0}}});
}

int main() {

  test0();

  return 0;
}
