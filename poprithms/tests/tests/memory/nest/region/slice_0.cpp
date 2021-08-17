// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/nest/region.hpp>

namespace {
using namespace poprithms::memory::nest;
void assertNonEmptySlice(const Region &r,
                         const std::vector<int64_t> &l,
                         const std::vector<int64_t> &u,
                         const Region &expected) {
  const auto slice = r.slice(l, u);
  if (slice.empty()) {
    throw poprithms::test::error("expected non-empty slice");
  }

  if (!slice.equivalent(expected)) {
    throw poprithms::test::error("Failed in assertSlice comparions");
  }
}

void assertEmptySlice(const Region &r,
                      const std::vector<int64_t> &l,
                      const std::vector<int64_t> &u,
                      const Shape &expected) {

  const auto sliced = r.slice(l, u);
  if (!sliced.empty()) {
    std::ostringstream oss;
    oss << "Failed empty slice test, sliced = " << sliced << ".";
    throw poprithms::test::error(oss.str());
  }

  if (expected.rank_u64() != l.size()) {
    throw poprithms::test::error("bad test, expected wrong size");
  }
  for (uint64_t i = 0; i < l.size(); ++i) {
    if (expected.dim(i) != u[i] - l[i]) {
      throw poprithms::test::error(
          "Failed shape comparison in empty slice test");
    }
  }
}
} // namespace

void test0() {
  Region r0 = Region::createFull({2, 3, 4});

  assertNonEmptySlice(
      r0, {0, 0, 0}, {1, 1, 1}, Region::createFull({1, 1, 1}));

  assertEmptySlice(r0, {0, 1, 0}, {1, 1, 1}, Shape{1, 0, 1});

  assertNonEmptySlice(
      r0, {1, 1, 3}, {2, 3, 4}, Region::createFull({1, 2, 1}));

  assertEmptySlice(r0, {1, 1, 1}, {1, 3, 4}, {0, 2, 3});
}

void test1() {
  Region r0({4, 6, 8}, {{{{1, 1, 0}}}, {{{1, 1, 0}}}, {{{1, 1, 0}}}});
  assertEmptySlice(r0, {1, 1, 1}, {2, 1, 2}, {1, 0, 1});
  assertEmptySlice(r0, {1, 1, 1}, {2, 2, 2}, {1, 1, 1});
  assertNonEmptySlice(
      r0,
      {0, 0, 0},
      {2, 2, 2},
      Region({2, 2, 2}, {{{{1, 1, 0}}}, {{{1, 1, 0}}}, {{{1, 1, 0}}}}));
  assertNonEmptySlice(
      r0,
      {0, 1, 2},
      {2, 3, 6},
      Region({2, 2, 4}, {{{{1, 1, 0}}}, {{{1, 1, 1}}}, {{{1, 1, 0}}}}));
}

void test2() {
  // xx...xx...
  auto S   = 10;
  auto on  = 2;
  auto off = 3;
  auto U   = 6;
  Region r0({S}, {{{{on, off, 0}}}});

  // xx...x
  auto b = r0.slice({0}, {U});

  std::vector<int64_t> expected{0, 1, 5};

  std::vector<int64_t> observed;
  for (auto reg : b.get()) {
    auto ons = reg.getOns()[0];
    observed.insert(observed.end(), ons.cbegin(), ons.cend());
  }
  std::sort(observed.begin(), observed.end());
  if (observed != expected) {
    throw poprithms::test::error(
        "Failure in slice test2, expected \nxx...xx...\nwhen sliced "
        "by interval [0,6) to be\nxx...x");
  }
}

void test3() {

  // reproducer of T44367.
  Region r0({131328}, {{{{256, 257, 0}}}});
  auto b = r0.slice({0}, {130816});
}

int main() {
  test0();
  test1();
  test2();
  test3();
}
