// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/sett.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms::memory::nest;
void assertMethod1(const Sett &x, const Sett &indices, uint64_t range) {

  //                              | "range" over which to test
  //
  // 1..11.1..11.1..11.1..11.1..11. "x"
  // 11...11...11...11...11...11... "indices"
  // 1.   .1   1.   11   .1   ..    "target" and hopefully "observed"

  // The indices at which to sample (0,1,5,6,10,11...)
  auto allIndices = indices.getOns(0, range);

  // bitmap for "x" as illustrated above
  std::vector<bool> xOns(range, false);
  for (const auto i : x.getOns(0, range)) {
    xOns[i] = true;
  }

  std::vector<bool> target;
  for (auto i : allIndices) {
    target.push_back(xOns[i]);
  }

  const auto sampled = x.sampleAt(indices);
  std::vector<bool> observed(target.size(), false);
  for (const auto &sett0 : sampled.get()) {
    for (auto i : sett0.getOns(0, target.size())) {
      observed[i] = true;
    }
  }

  if (target != observed) {
    std::ostringstream oss;
    oss << "Failure in assertMethod1, where x = " << x
        << ", indices = " << indices << " and range = " << range;
    throw error(oss.str());
  }
}

void assertMethod2(const Sett &x,
                   const Sett &indices,
                   const std::vector<int> &expected,
                   uint64_t repls = 2) {

  uint64_t nChecks = repls * expected.size();
  std::vector<int> observed(nChecks, 0);
  const auto sampled = x.sampleAt(indices);
  for (const auto &sett0 : sampled.get()) {
    for (auto i : sett0.getOns(0, nChecks)) {
      ++observed[i];
    }
  }

  std::vector<int> replExpected(nChecks);
  for (uint64_t i = 0; i < repls; ++i) {
    for (uint64_t j = 0; j < expected.size(); ++j) {
      replExpected[i * expected.size() + j] = expected[j];
    }
  }

  if (replExpected != observed) {
    std::ostringstream oss;
    oss << "Failure un assertMethod2 with x = " << x
        << ", indices = " << indices << " expected = ";
    poprithms::util::append(oss, expected);
    oss << " and repls = " << repls;
    throw error(oss.str());
  }
}

} // namespace

int main() {

  // 11..  x
  // 11..  indices
  // 11    answer
  assertMethod2({{{2, 2, 0}}}, {{{2, 2, 0}}}, {1, 1});

  // 1.1.1.   x
  // 1.11.1   indices
  // 1 1. .   answer
  assertMethod2({{{1, 1, 0}}}, {{{2, 1, 2}}}, {1, 1, 0, 0});

  // 1.1  x
  // 11.  indices
  // 1.   answer
  assertMethod2({{{2, 1, 2}}}, {{{2, 1, 0}}}, {1, 0});

  // 11.  x
  // .11  indices
  //  1.  answer
  assertMethod2({{{2, 1, 0}}}, {{{2, 1, 1}}}, {1, 0});

  // .1111111..  x
  // 111......1  indices
  // .11      .  answer
  assertMethod2({{{7, 3, 1}}}, {{{4, 6, 9}}}, {0, 1, 1, 0});

  // 1111....1111111  x
  // ..11111........  indices
  //   11...
  assertMethod2({{{11, 4, 8}}}, {{{5, 10, 2}}}, {1, 1, 0, 0, 0});

  // 1111....1111111  x
  // 11111..........  indices
  // 1111.            answer
  assertMethod2({{{11, 4, 8}}}, {{{5, 10, 0}}}, {1, 1, 1, 1, 0});

  // 1.1.1.  x
  // 11.11.  indices
  // 1. .1   answer
  assertMethod2({{{1, 1, 0}}}, {{{2, 1, 0}}}, {1, 0, 0, 1});

  // 1.1.1.  x
  // .11.11  indices
  //  .1 1.  answer
  assertMethod2({{{1, 1, 0}}}, {{{2, 1, 1}}}, {0, 1, 1, 0});

  // .1.1.1  x
  // 1.11.1  indices
  // . .1 1  answer
  assertMethod2({{{1, 1, 1}}}, {{{2, 1, 2}}}, {0, 0, 1, 1});

  // .1.1.1.1.1.1.1  x
  // 11111.111111.1  indices
  // .1.1. .1.1.1 1  answer
  assertMethod2(
      {{{1, 1, 1}}}, {{{6, 1, 6}}}, {0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1});

  // .1.1.1.1.1.1.1  x
  // 11....111....1  indices
  // .1    .1.    1  answer

  assertMethod2({{{1, 1, 1}}}, {{{3, 4, 6}}}, {0, 1, 0, 1, 0, 1});

  // .1.1.1.1.1.1.1  x
  // ..111....111..  indices
  //   .1.    1.1    anwer

  assertMethod2({{{1, 1, 1}}}, {{{3, 4, 2}}}, {0, 1, 0, 1, 0, 1});

  // 1.1.1.1.1.1.1.1.1.1.1.  x
  // ..1111111....1111111..  indices
  //   1.1.1.1    .1.1.1.    answer
  assertMethod2({{{1, 1, 0}}},
                {{{7, 4, 2}}},
                {1,
                 0,
                 1,
                 0,
                 1,
                 0,
                 1, //
                 0,
                 1,
                 0,
                 1,
                 0,
                 1,
                 0});

  // 11.11.11.11.11.11.11.11.11.11.11   x
  // ..1111111....1111111....1111111..  indices
  //   .11.11.    1.11.11    11.11.1    answer

  assertMethod2({{{2, 1, 0}}}, {{{7, 4, 2}}}, {0, 1, 1, 0, 1, 1, 0, //
                                               1, 0, 1, 1, 0, 1, 1, //
                                               1, 1, 0, 1, 1, 0, 1});

  // 11.11.11.11.11.11.11.11.11.11.11   x
  // 1111111....1111111....1111111....  indices
  // 11.11.1    .11.11.    1.11.11      answer

  assertMethod2({{{2, 1, 0}}}, {{{7, 4, 0}}}, {1, 1, 0, 1, 1, 0, 1, //
                                               0, 1, 1, 0, 1, 1, 0, //
                                               1, 0, 1, 1, 0, 1, 1}

  );

  // 11..11..11..11..11..11..11..11..11..11..11..  x
  // ..1111111....1111111....1111111....1111111..  indices
  //   ..11..1    1..11..    11..11.    .11..11    answer

  assertMethod2({{{2, 2, 0}}}, {{{7, 4, 2}}}, {0, 0, 1, 1, 0, 0, 1, //
                                               1, 0, 0, 1, 1, 0, 0, //
                                               1, 1, 0, 0, 1, 1, 0, //
                                               0, 1, 1, 0, 0, 1, 1});

  assertMethod1({{{4, 2, 0}, {3, 0, 0}, {1, 1, 0}}}, {{{1, 0, 0}}}, 12);

  // 0123456789012345678901234567890123
  // ...1.1................1.1.          x
  // 11.111.111.........11.111.111.....  indices
  // .. 1.1 ...         .. 1.1 ...       answer

  //(((3,16,3)(1,1,0)), ((10,9,0)(3,1,3)))
  Sett x{{{3, 16, 3}, {1, 1, 0}}};
  Sett filter{{{10, 9, 0}, {3, 1, 3}}};

  //(((3,16,3)(1,1,0)), ((10,9,0)(3,1,3)))
  //

  // (((3,7,3)(1,1,0)), ((3,1,3)))
  // ...x.x.......x.x               x
  // xx.xxx.xxx.xxx.xxx.            indices

  std::cout << "\n\n\n\n\n\n\nfootest\n\n\n\n\n\n" << std::endl;
  auto z = x.sampleAt(filter);
  assertMethod1(x, filter, 100);
  std::cout << "z = " << z << std::endl;
}
