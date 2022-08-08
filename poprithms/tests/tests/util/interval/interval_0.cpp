// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/util/interval.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/stridedpartition.hpp>

namespace {
using namespace poprithms::util;

void test0() {

  // A bunch of Intervals which should all be the same once canonicalized
  // (Individual Intervals sorted, intersected, etc.)
  std::vector<Intervals> i0;
  i0.push_back(Intervals({{3, 4}, {6, 7}, {7, 8}, {8, 9}}));
  i0.push_back(Intervals({{3, 4}, {6, 9}}));
  i0.push_back(Intervals({{6, 9}, {3, 4}}));
  i0.push_back(Intervals({{8, 9}, {6, 8}, {3, 4}}));
  i0.push_back(Intervals({{8, 9}, {6, 8}, {3, 4}, {8, 9}}));
  i0.push_back(Intervals({{8, 9}, {6, 8}, {3, 4}, {7, 8}}));
  i0.push_back(Intervals({{7, 8}, {8, 9}, {6, 8}, {3, 4}, {7, 8}}));

  // This Intervals is different to the above ones, it doesn't contain 6.
  Intervals other({{3, 4}, {7, 9}});

  for (auto i : i0) {

    auto baseString = [&i]() {
      return std::string("Failure in test0 with Interval i = ") + i.str() +
             ". ";
    };
    if (i != i0[0]) {
      throw poprithms::test::error(
          baseString() + " This differs from i0[0], " + i0[0].str() +
          " but all Intervals in i0 are identical. ");
    }
    if (i == other) {
      throw poprithms::test::error(
          baseString() + " should not be the same as other, " + other.str());
    }

    if (i.size() != 4) {
      throw poprithms::test::error(baseString() +
                                   ": All Intervals of size 4");
    }

    if (i.contiguousFromZero()) {
      throw poprithms::test::error(baseString() +
                                   ": Not contiguous from zero");
    }
  }
}

void test1() {
  // Test the application of `Interval::subIntervals` on an Interval created
  // from `{{0, 1}, {10, 12}, {20, 23}, {30, 34}}` yields specific intervals
  // for different values of `r0` and `r1`.
  //

  // [0,1) is of size 1
  // [10,12) is of size 2
  // [20, 23) is of size 3
  // [30, 34) is of size 4
  // So the total size if is (below) is 1 + 2 + 3 + 4 = 10.
  Intervals is({{0, 1}, {10, 12}, {20, 23}, {30, 34}});
  auto assertCorrect =
      [is](int64_t r0, int64_t r1, const Intervals &expected) {
        if (is.subIntervals(r0, r1) != expected) {
          std::ostringstream ost;
          ost << "Failed to assert correct Intervals. For is = " << is
              << ", expected is.subIntervals(r0 = " << r0 << ", r1 = " << r1
              << ") to be " << expected << ", not "
              << is.subIntervals(r0, r1);
          throw poprithms::test::error(ost.str());
        }
      };

  assertCorrect(0, 10, is);
  assertCorrect(1, 10, Intervals({{10, 12}, {20, 23}, {30, 34}}));
  assertCorrect(2, 10, Intervals({{11, 12}, {20, 23}, {30, 34}}));
  assertCorrect(3, 10, Intervals({{20, 23}, {30, 34}}));
  assertCorrect(4, 10, Intervals({{21, 23}, {30, 34}}));
  assertCorrect(5, 10, Intervals({{22, 23}, {30, 34}}));
  assertCorrect(6, 10, Intervals({{30, 34}}));
  assertCorrect(9, 10, Intervals({{33, 34}}));
  assertCorrect(10, 10, Intervals({}));
  assertCorrect(10, 10, Intervals({100, 100}));

  assertCorrect(0, 9, Intervals({{0, 1}, {10, 12}, {20, 23}, {30, 33}}));
  assertCorrect(0, 8, Intervals({{0, 1}, {10, 12}, {20, 23}, {30, 32}}));
  assertCorrect(0, 7, Intervals({{0, 1}, {10, 12}, {20, 23}, {30, 31}}));
  assertCorrect(0, 6, Intervals({{0, 1}, {10, 12}, {20, 23}}));
  assertCorrect(0, 5, Intervals({{0, 1}, {10, 12}, {20, 22}}));
  assertCorrect(0, 4, Intervals({{0, 1}, {10, 12}, {20, 21}}));
  assertCorrect(0, 3, Intervals({{0, 1}, {10, 12}}));

  assertCorrect(1, 3, Intervals({{10, 12}}));
  assertCorrect(2, 3, Intervals({{11, 12}}));
  assertCorrect(3, 3, Intervals({}));

  assertCorrect(100, -100, Intervals({}));
  assertCorrect(-100, +100, is);
}

void test2() {
  // Test of empty intervals
  Interval a(3, 3);
  if (a.size() != 0) {
    throw poprithms::test::error("a is an empty Interval");
  }
  Intervals b(3, 3);
  if (b.size() != 0) {
    throw poprithms::test::error("b is an empty Intervals");
  }
  Intervals c({{1, 1}, {10, 10}, {5, 5}});
  if (c.size() != 0) {
    throw poprithms::test::error("c is an empty Intervals");
  }
  Intervals d({{1, 1}, {10, 10}, {6, 7}, {5, 5}});
  if (d.size() != 1) {
    throw poprithms::test::error("d contains 1 element");
  }
}

template <typename X>
std::ostream &operator<<(std::ostream &ost, std::vector<X> &xs) {
  std::vector<std::string> strings;
  for (auto x : xs) {
    strings.push_back(poprithms::util::getStr(x));
  }
  poprithms::util::append(ost, strings);
  return ost;
}

void testStridedInterval0() {
  StridedPartition sp(18, 3, 2);
  auto gs = sp.groups();
  // 0 1 0 1 0 1 2 3 2 3 2 3 4 5 4 5 4 5
  std::vector<std::vector<uint64_t>> expected{{0, 2, 4},
                                              {1, 3, 5},
                                              {6, 8, 10},
                                              {7, 9, 11},
                                              {12, 14, 16},
                                              {13, 15, 17}};

  if (sp.group(16) != 4) {
    std::ostringstream oss;
    oss << "Expected index 16 to be in group 4:"
        << "\n0 1 0 1 0 1 2 3 2 3 2 3 4 5 4 5 4 5"
        << "\n                               ^^^ "
        << "\n0 1 2 ...                      16  "
        << "\nnot " << sp.group(16);
    throw poprithms::test::error(oss.str());
  }

  if (gs != expected) {
    std::ostringstream oss;
    oss << "For StridedPartition " << sp << ", expected groups to be "
        << expected << ", but it was " << gs;
    throw poprithms::test::error(oss.str());
  }

  if (sp.nGroups() != 6) {
    throw poprithms::test::error("There are 6 groups");
  }

  if (sp.indicesInGroup(3) != std::vector<uint64_t>{7, 9, 11}) {
    throw poprithms::test::error("Indices in group #2 are 7,9, and 11");
  }

  bool caught{false};
  try {
    StridedPartition(19, 3, 2);
  } catch (const poprithms::error::error &) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error(
        "Failed to catch incompatible stride partition parameters");
  }
}

} // namespace

int main() {

  test0();
  test1();
  test2();
  testStridedInterval0();

  return 0;
}
