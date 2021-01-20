// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/memory/chain/chain.hpp>
#include <poprithms/memory/chain/error.hpp>

namespace {

using namespace poprithms::memory::chain;

// Multiple chained Reshapes. Expect them to be collapsed into a single
// Reshape.
void testMergeCommonReshape() {

  Chain chain({12, 13});
  chain.slice({1, 2}, {11, 12})
      .reshape({5, 20})
      .reshape({20, 5})
      .reshape({1, 1, 100});
  auto merged = chain.mergeContiguousSameType();

  Chain expected({12, 13});
  expected.slice({1, 2}, {11, 12}).reshape({1, 1, 100});
  merged.confirmEqual(expected);
  chain.confirmNotEqual(expected);
}

// Multiple chained DimShuffles. Expect them to be collapsed into a single
// DimShuffle
void testMergeCommonDimShuffle() {
  Chain chain({2, 3, 5, 7});
  chain.dimShuffle({{1, 2, 3, 0}}).dimShuffle({{1, 2, 3, 0}}).flatten();
  const auto merged = chain.mergeContiguousSameType();

  Chain expected({2, 3, 5, 7});
  expected.dimShuffle({{2, 3, 0, 1}}).flatten();
  merged.confirmEqual(expected);
}

void testMergeCommonReverse() {

  Chain chain({2, 3, 5});
  chain.reverse(Dimensions({1, 2}))
      .reverse(Dimensions({0, 0, 1}))
      .reverse(Dimensions({0, 1, 2}))
      .reverse(Dimensions({1}))
      .flatten();
  const auto merged = chain.mergeContiguousSameType();

  // counts
  // 0 : 3
  // 1 : 4
  // 2 : 2
  // Only index 0 has an odd number of reversals.
  const auto expected = Chain({2, 3, 5}).reverse(Dimensions({0})).flatten();
  merged.confirmEqual(expected);
}

void testMergeCommonSettSample0() {
  Chain chain({100});
  chain.slice({10}, {90});
  chain.subSample(Stride(2), Dimension(0));
  auto merged = chain.mergeContiguousSameType();
  Chain expected({100});
  expected.settSample({{{{80, 20, 10}, {1, 1, 0}}}});
  merged.confirmEqual(expected);
}

void testMergeCommonSettSample1() {
  {
    Chain chain({100});
    chain.settSample({{{{10, 10, 0}}}}).settSample({{{{3, 3, 0}}}});
    const auto merged = chain.mergeContiguousSameType();
    chain.confirmEqual(merged);
  }

  {
    Chain chain({100});
    chain.settSample({{{{10, 10, 0}}}}).settSample({{{{3, 2, 0}}}});
    const auto merged = chain.mergeContiguousSameType();
    chain.confirmNotEqual(merged);
  }
}

void testMergeCommonSettSample2() {
  Chain chain({200 * 17});

  // These 2 are mergeable (as 100 mod (19 + 6) == 0)
  chain.settSample({{{{100, 100, 0}}}}).settSample({{{{19, 6, 0}}}});

  // These 2 are mergeable (as 10 mod (4 + 1) == 0)
  chain.settSample({{{{10, 7, 0}}}}).settSample({{{{4, 1, 2}}}});

  // But 25 mod 17 is not 0, so trying to fill {10,7,0} into {19,6,0} will
  // result in a shattering into multiple Regions.

  chain.mergeContiguousSameType().confirmEqual(
      Chain({200 * 17})
          .settSample({{{{100, 100, 0}, {19, 6, 0}}}})
          .settSample({{{{10, 7, 0}, {4, 1, 2}}}}));
}

void testMergeCommonSettSample3() {
  Chain({97})
      .subSample(Stride(2), Dimension(0))
      .subSample(Stride(3), Dimension(0))
      .subSample(Stride(5), Dimension(0))
      .mergeContiguousSameType()
      .confirmEqual(Chain({97}).subSample(Stride(30), Dimension(0)));
}

void testMergeCommonSettFillInto0() {
  Chain({100})
      .settFillInto(Stride(2), Dimension(0))
      .settFillInto(Stride(3), Dimension(0))
      .settFillInto(Stride(5), Dimension(0))
      .mergeContiguousSameType()
      .confirmEqual(
          Chain({100}).settFillInto(Stride(2 * 3 * 5), Dimension(0)));
}

void testMergeCommonSettFillInto1() {
  // unmergeable settFillInto:
  auto chain = Chain({70})
                   .settFillInto(Region({180}, {{{{7, 11, 0}}}}))
                   .settFillInto(Region({(180 / 13) * 30 + 180 % 13},
                                        {{{{13, 17, 0}}}}));
  chain.mergeContiguousSameType().confirmEqual(chain);
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
  return 0;
}
