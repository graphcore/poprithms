// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <iostream>

#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/region.hpp>

namespace {
using namespace poprithms::memory::nest;
void assertNonEmptySlice(const Region &r,
                         const std::vector<int64_t> &l,
                         const std::vector<int64_t> &u,
                         const Region &expected) {
  const auto slice = r.slice(l, u);
  if (slice.empty()) {
    throw error("expected non-empty slice");
  }

  if (!slice.equivalent(expected)) {
    throw error("Failed in assertSlice comparions");
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
    throw error(oss.str());
  }

  if (expected.rank_u64() != l.size()) {
    throw error("bad test, expected wrong size");
  }
  for (uint64_t i = 0; i < l.size(); ++i) {
    if (expected.dim(i) != u[i] - l[i]) {
      throw error("Failed shape comparison in empty slice test");
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

int main() {
  test0();
  test1();
}
