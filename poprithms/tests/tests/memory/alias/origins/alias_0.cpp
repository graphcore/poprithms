// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/alias/origins.hpp>

namespace {
using namespace poprithms::memory::alias;
using namespace poprithms::memory::nest;

void test0() {
  Origins o({10, 2});
  o.insert({4}, {Region::createFull({11, 1})});
  o.insert({2}, {Region::createFull({1, 2, 2})});
  o.insert({2}, {Region::createFull({1, 2, 2})});

  if (!o.containsAliases()) {
    throw poprithms::test::error(
        std::string("Only 19 elements in Origins for ") +
        "a 20 element Shape impossible to have aliases");
  }

  o.insert({10}, {Region::createFull({1, 1, 1, 1})});
  // Total elements = 3*4 + 3*1*1 + 1*2*2 + 1*1*1*1 = 20.

  const auto allocIds = o.getAllocIds();
  if (allocIds.size() != 3) {
    throw poprithms::test::error("Expected 3 elements in test0 Origins");
  }
  for (auto i : {2, 4, 10}) {
    if (std::find(std::cbegin(allocIds), std::cend(allocIds), AllocId(i)) ==
        std::cend(allocIds)) {
      throw poprithms::test::error("Expected " + std::to_string(i) +
                                   " to be an alloc");
    }
  }

  if (!o.containsAliases()) {
    throw poprithms::test::error(
        "All 20 elements have separate allocation addresses, but "
        "Allocation 2 has aliases");
  }
}

void test1() {

  Origins o({8, 4});
  o.insert({0}, Region::createFull({4}));
  o.clear();
  o.insert({1}, Region::fromStripe({2, 16}, 0, {1, 1, 0}));
  o.insert({1}, Region::fromStripe({2, 16}, 0, {1, 1, 1}));
  if (o.containsAliases()) {
    throw poprithms::test::error(
        "The 2 stripes do not intersect and together have 32 elements");
  }
}

void test2() {
  Origins o0({5, 10});
  Origins o1({5, 10});

  o0.insert({10}, Region::createEmpty({5, 10}));
  o1.insert({20}, Region::createEmpty({5, 10}));

  o0.insert({1}, Region::createFull({5, 10}));
  o1.insert({1}, Region::createFull({5, 10}));

  if (!o0.contains(o1) || !o1.contains(o0)) {
    throw poprithms::test::error(
        "Empty regions should have not effect in checking for subsets");
  }

  o0.insert({3}, Region::createFull({5, 10}).slice({0, 0}, {1, 1}));

  if (!o0.contains(o1) || o1.contains(o0)) {
    throw poprithms::test::error(
        "o0 contains o1, but the reverse is not true");
  }
}

} // namespace

int main() {

  test0();
  test1();
  test2();

  return 0;
}
