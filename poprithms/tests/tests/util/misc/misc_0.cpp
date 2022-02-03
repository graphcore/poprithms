// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <vector>

#include <poprithms/error/error.hpp>
#include <poprithms/util/contiguoussubset.hpp>
#include <poprithms/util/typedinteger.hpp>

namespace {

using namespace poprithms::util;

void test0() {

  ContiguousSubset<int> foo(5, {0, 2, 4});
  if (foo.inFullset(1) != 3) {
    throw poprithms::test::error(
        "element #1 in the subset is elements #3 in the full set");
  }
  if (foo.inSubset(1) != 0) {
    throw poprithms::test::error(
        "element #1 in the full set is elements #0in the subset");
  }
  if (!foo.isRemoved(0) || foo.isRemoved(1)) {
    throw poprithms::test::error("0 is removed, 1 is not");
  }
  if (foo.nSubset() != 2) {
    throw poprithms::test::error("the full set has 5 elements, 3 were "
                                 "removed, so there are 2 remaining");
  }
  std::vector<std::string> x{"a", "b", "c", "d", "e"};
  foo.reduce(x);
  if (x != std::vector<std::string>{"b", "d"}) {
    throw poprithms::test::error("elements 1 and 3 are \"b\" and \"d\"");
  }
}

void test1() {
  ContiguousSubset<TypedInteger<'c', int>> x(10, {1, 2, 3, 4});
  if (!x.isRemoved(1) || (x.isRemoved(7))) {
    throw poprithms::test::error("TypedInteger 1 is removed, and 7 is not");
  }
}

} // namespace

int main() {
  test0();
  test1();
  return 0;
}
