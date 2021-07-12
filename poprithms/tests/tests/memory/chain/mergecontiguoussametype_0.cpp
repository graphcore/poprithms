// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/chain/chain.hpp>

namespace {

using namespace poprithms::memory::chain;

// Multiple chained Reshapes. Expect them to be collapsed into a single
// Reshape.
void testMergeCommonReshape() {

  Chain chain({12, 13});
  chain.slice({1, 2}, {11, 12});
  chain.reshape({5, 20});
  chain.reshape({20, 5});
  chain.reshape({1, 1, 100});

  Chain expected({12, 13});
  expected.slice({1, 2}, {11, 12});
  expected.reshape({1, 1, 100});
  chain.canonicalized().confirmEqual(expected);
}

// Multiple chained DimShuffles. Expect them to be collapsed into a single
// DimShuffle
void testMergeCommonDimShuffle() {
  Chain chain({2, 3, 5, 7});
  chain.dimShuffle({{1, 2, 3, 0}});
  chain.dimShuffle({{1, 2, 3, 0}});
  chain.flatten();
  const auto merged = chain.canonicalized();

  Chain expected({2, 3, 5, 7});
  expected.dimShuffle({{2, 3, 0, 1}});
  expected.flatten();

  merged.confirmEqual(expected);
}

void testMergeCommonReverse() {

  Chain chain({2, 3, 5});
  chain.reverse(Dimensions({1, 2}));
  chain.reverse(Dimensions({0, 0, 1}));
  chain.reverse(Dimensions({0, 1, 2}));
  chain.reverse(Dimensions({1}));
  chain.flatten();
  const auto merged = chain.canonicalized();

  // counts
  // 0 : 3
  // 1 : 4
  // 2 : 2
  // Only index 0 has an odd number of reversals.
  auto expected = Chain({2, 3, 5});
  expected.reverse(Dimensions({0}));
  expected.flatten();
  merged.confirmEqual(expected);
}

void testMergeCommonSettSample0() {
  Chain chain({100});
  chain.slice({10}, {90});
  chain.subSample(Stride(2), Dimension(0));
  auto merged = chain.canonicalized();
  Chain expected({100});
  expected.settSample({{{{80, 20, 10}, {1, 1, 0}}}});
  merged.confirmEqual(expected);
}

void testMergeCommonSettSample1() {
  {
    Chain chain({100});
    chain.settSample({{{{10, 10, 0}}}});
    chain.settSample({{{{3, 3, 0}}}});
    const auto merged = chain.canonicalized();
    chain.confirmEqual(merged);
  }

  {
    Chain chain({100});
    chain.settSample({{{{10, 10, 0}}}});
    chain.settSample({{{{3, 2, 0}}}});
    const auto merged = chain.canonicalized();
    chain.confirmNotEqual(merged);
  }
}

void testMergeCommonSettSample2() {
  Chain chain({200 * 17});

  // These 2 are mergeable (as 100 mod (19 + 6) == 0)
  chain.settSample({{{{100, 100, 0}}}});
  chain.settSample({{{{19, 6, 0}}}});

  // These 2 are mergeable (as 10 mod (4 + 1) == 0)
  chain.settSample({{{{10, 7, 0}}}});
  chain.settSample({{{{4, 1, 2}}}});

  // But 25 mod 17 is not 0, so trying to fill {10,7,0} into {19,6,0} will
  // result in a shattering into multiple Regions.

  Chain expected({200 * 17});
  expected.settSample({{{{100, 100, 0}, {19, 6, 0}}}});
  expected.settSample({{{{10, 7, 0}, {4, 1, 2}}}});
  chain.canonicalized().confirmEqual(expected);
}

void testMergeCommonSettSample3() {

  Chain expected({97});
  expected.subSample(Stride(30), Dimension(0));

  Chain chain({97});
  chain.subSample(Stride(2), Dimension(0));
  chain.subSample(Stride(3), Dimension(0));
  chain.subSample(Stride(5), Dimension(0));
  chain.canonicalized().confirmEqual(expected);
}

void testMergeCommonSettFillInto0() {

  Chain expected({100});
  expected.settFillInto(Stride(2 * 3 * 5), Dimension(0));

  Chain c({100});
  c.settFillInto(Stride(2), Dimension(0));
  c.settFillInto(Stride(3), Dimension(0));
  c.settFillInto(Stride(5), Dimension(0));
  c.canonicalized().confirmEqual(expected);
}

void testMergeCommonSettFillInto1() {
  // unmergeable settFillInto:
  Chain chain({70});
  chain.settFillInto(Region({180}, {{{{7, 11, 0}}}}));
  chain.settFillInto(Region({(180 / 13) * 30 + 180 % 13}, {{{{13, 17, 0}}}}));
  chain.canonicalized().confirmEqual(chain);
}

void testMergeCommonSettFillInto2() {
  Chain chain({70});
  const auto r0 = Region({180}, {{{{7, 11, 0}}}});
  chain.settFillInto(r0);
  const auto r1 = Region({(180 / 13) * 30 + 180 % 13}, {{{{13, 17, 0}}}});
  chain.settFillInto(r1);

  // unmergeable settFillInto:
  chain.canonicalized().confirmEqual(chain);

  // now, if we add a reverse on the end, we should be able to bubble it back
  // through both of the SettFills.
  chain.reverse(Dimensions{0});
  auto canon = chain.canonicalized();

  Chain expected({70});
  expected.reverse(Dimension(0));
  expected.settFillInto(r0.reverse({0}));
  expected.settFillInto(r1.reverse({0}));
  expected.confirmEqual(canon);
}

} // namespace

int main() {
  testMergeCommonReshape();
  testMergeCommonDimShuffle();
  testMergeCommonReverse();
  testMergeCommonSettSample0();
  testMergeCommonSettSample1();
  testMergeCommonSettSample2();
  testMergeCommonSettSample3();
  testMergeCommonSettFillInto0();
  testMergeCommonSettFillInto1();
  testMergeCommonSettFillInto2();
  return 0;
}
