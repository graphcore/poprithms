// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <memory/chain/op.hpp>
#include <poprithms/memory/chain/chain.hpp>
#include <poprithms/memory/chain/error.hpp>

namespace {

using namespace poprithms::memory::chain;

void testCanonicalize0() {
  // A few passes of canonicalization, and this Chain is seen to be the
  // identity Chain.
  //
  //
  // First, merging contiguous same type produces:
  //   dimShuffle({2,3,0,1})
  //   reshape({6,7,4,5})
  //   dimShuffle({2,3,0,1})
  //
  // Then, reshape({6,7,4,5}) is seen to be identity, reducing to:
  //   dimShuffle({2,3,0,1})
  //   dimShuffle({2,3,0,1})
  //
  // which is the merged into dimShuffle({0,1,2,3}), which is the identity.
  //
  Chain a({4, 5, 6, 7});
  a.dimShuffle({{1, 2, 3, 0}});
  a.dimShuffle({{1, 2, 3, 0}});
  a.reshape({20, 42});
  a.reshape({6, 7, 4, 5});
  a.dimShuffle({{1, 2, 3, 0}});
  a.dimShuffle({{1, 2, 3, 0}});
  a.canonicalized().confirmEqual(Chain({4, 5, 6, 7}));
}

void testMapToEmpty() {

  Chain c({10});
  c.mask(Region::fromStripe({10}, 0, {1, 2, 0}));
  c.mask(Region::fromStripe({10}, 0, {1, 2, 1}));
  c.canonicalize();
  if (c.nOps() > 2) {
    throw error("This Chain is empty. the full Region gets mapped to empty "
                "Region. This can be represented with 2 Ops");
  }
}

void testBubbleReverseDimShuffle() {

  //  (2,3,5,7) ----> Reverse((0))
  //                  DimShuffle((1,2,3,0))  ----> (3,5,7,2)
  //
  //  (2,3,5,7) ----> DimShuffle((1,2,3,0))
  //                  Reverse((3))           ----> (3,5,7,2)

  Chain c({2, 3, 5, 7});
  c.reverse(Dimensions({0}));
  c.dimShuffle({{1, 2, 3, 0}});

  c.canonicalize();

  Chain expected({2, 3, 5, 7});
  expected.dimShuffle({{1, 2, 3, 0}});
  expected.reverse(Dimensions({3}));

  expected.confirmEqual(c);
}

void testBubbleDimShuffleReverse() {

  //  (2,3,5,7) ----> DimShuffle((1,2,3,0))
  //                  Reverse((3))           ----> (3,5,7,2)

  Chain c({2, 3, 5, 7});
  c.dimShuffle({{1, 2, 3, 0}});
  c.reverse(Dimensions({3}));
  // There should be no change, as DimShuffle appears before Reverse
  // lexicographically.
  c.canonicalized().confirmEqual(c);

  const Shape inShape0{3, 5, 2};
  const Permutation p({1, 2, 0});
  Op x0(Type::DimShuffle, inShape0.dimShuffle(p), p);
  Op x1(Type::Reverse, {2, 3, 5}, Dimensions({0}));

  auto swapped = Op::bubbleReverseBack(inShape0, x0, x1);
  if (!swapped) {
    throw error("Failed to swap reverse and dimShuffle");
  }
  if (x0.type() != Type::Reverse) {
    throw error("x0 and x1 should have had their types swapped");
  }
  if (x0.attr().dimensions() != Dimensions({1})) {
    std::ostringstream oss;
    oss << "Before the swap, dimension 0 was reversed after the permutation"
        << " [1 2 0]"
        << ". dimension 0 after the permutation corresponds to dimension 1"
        << " before the permutation. Therefore expected the Dimensions of "
        << "the Reverse before the DimShuffle to be {1}.";
    throw error(oss.str());
  }
}

void testBubbleSettSampleDimShuffle() {

  Chain c({20, 30, 50});
  c.slice({0, 0, 0}, {15, 25, 45});
  c.dimShuffle({{1, 2, 0}});

  c.canonicalize();

  Chain expected({20, 30, 50});
  expected.dimShuffle({{1, 2, 0}});
  expected.slice({0, 0, 0}, {25, 45, 15});

  expected.confirmEqual(c);
}

void testBubbleSettSampleReverse() {

  Chain chain({10});
  chain.slice({7}, {10});
  chain.reverse(Dimensions({0}));
  chain.canonicalize();

  Chain expected({10});
  expected.reverse(Dimensions({0}));
  expected.slice({0}, {3});

  expected.confirmEqual(chain);
}

void rubixTwist() {

  // 012
  // 345
  Chain chain({2, 3});

  // 345
  // 012
  chain.reverse(Dimension(0));

  // 30
  // 41
  // 52
  chain.dimShuffle({{1, 0}});

  // 03
  // 14
  // 25
  chain.reverse(Dimension(1));

  // 012
  // 345
  chain.dimShuffle({{1, 0}});

  // Chain does nothing, it is identity!

  chain.canonicalize();

  Chain({2, 3}).confirmEqual(chain);
}

} // namespace

int main() {
  testCanonicalize0();
  testMapToEmpty();
  testBubbleReverseDimShuffle();
  testBubbleSettSampleDimShuffle();
  testBubbleSettSampleReverse();
  testBubbleDimShuffleReverse();
  rubixTwist();
  return 0;
}
