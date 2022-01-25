// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/chain/chain.hpp>

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
    throw poprithms::test::error(
        "This Chain is empty. the full Region gets mapped to empty "
        "Region. This can be represented with 2 Ops");
  }
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

  chain.canonicalize();

  // Chain does nothing, it is identity when canonicalized!
  Chain({2, 3}).confirmEqual(chain);
}

void testBubbleSettSampleReverse0() {

  Chain chain({10});
  chain.slice({7}, {10});
  chain.reverse(Dimensions({0}));
  chain.canonicalize();

  Chain expected({10});
  expected.reverse(Dimensions({0}));
  expected.slice({0}, {3});

  expected.confirmEqual(chain);
}

void testRedundantSampleFill0() {

  const Region r0({15}, {{{{5, 10, 5}}}});
  const Region r1({15}, {{{{3, 12, 6}}}});
  const Region r2({15}, {{{{7, 8, 4}}}});

  {
    // In this case the sampling might eliminate some elements as r1 does not
    // contain r0. So the canonicalization pass cannot eliminate the final 2
    // ops.
    Chain chain({5});
    chain.settFillInto(r0);
    chain.settSample(r1);
    chain.settFillInto(r1);
    chain.canonicalized().confirmEqual(chain);
  }

  {
    // In this case the sampling cannot eliminate any elements, as r2 does
    // contain r0. So the final 2 ops can be elimininated.
    Chain chain({5});
    chain.settFillInto(r0);
    auto expected = chain;
    chain.settSample(r2);
    chain.settFillInto(r2);
    chain.canonicalized().confirmEqual(expected);
  }
}

void testRedundantSampleFill1() {

  using poprithms::memory::nest::Sett;

  // [11]
  Chain chain({2});

  // [11000]
  chain.settFillInto(Region({5}, {{{{2, 3, 0}}}}));

  const auto chain0 = chain;
  // These 2 links in the chain have no effect:
  {
    // 11..0 -> [011]
    chain.settSample(Region({5}, {{{{3, 2, 4}}}}));
    // [11000]
    chain.settFillInto(Region({5}, {{{{3, 2, 4}}}}));
  }

  chain.confirmNotEqual(chain0);
  chain.canonicalize();
  chain.confirmEqual(chain0);
}

void testExpandDimshuffle0() {
  Chain c({3});
  c.reshape({1, 3});
  c.expand({2, 3});
  c.dimShuffle({{1, 0}});

  Chain expected({3});
  expected.reshape({1, 3});
  expected.dimShuffle({{1, 0}});
  expected.expand({3, 2});
  c.canonicalized().confirmEqual(expected.canonicalized());
}

void testExpandReverse0() {

  Chain c0({4, 1, 5, 1});
  c0.reverse(Dimensions({0, 3, 2}));
  c0.expand({4, 7, 5, 8});

  Chain c1({4, 1, 5, 1});
  c1.expand({4, 1, 5, 8});
  c1.expand({4, 7, 5, 8});
  c1.reverse(Dimensions({0, 2, 3}));

  c0.canonicalized().confirmEqual(c1.canonicalized());
}

void testExpandSettSample0() {

  {
    Chain c0({5, 1, 7, 2});
    c0.slice({0, 0, 0, 0}, {5, 1, 3, 2});
    c0.expand({5, 8, 3, 2});

    Chain c1({5, 1, 7, 2});
    c1.expand({5, 8, 7, 2});
    c1.slice({0, 0, 0, 0}, {5, 8, 3, 2});

    c0.canonicalized().confirmEqual(
        c1.canonicalized(),
        "As the expansion dimension is 1 before the slice, the expansion and "
        "slice are permutable");
  }

  {
    Chain c0({4, 3});
    c0.slice({0, 0}, {1, 3});
    c0.expand({7, 3});

    auto c1      = c0.canonicalized();
    auto slices  = c1.where(Type::SettSample);
    auto expands = c1.where(Type::Expand);
    if (slices.size() != 1 || expands.size() != 1 || expands[0] < slices[0]) {
      throw poprithms::test::error(
          "Expected 1 slice, appearing after 1 expand. As the expansion "
          "dimension (0) is not of size 1 before the slice, the expand and "
          "slice cannot be permuted");
    }
  }
}

void testExpandReshape0() {
  {

    Chain c0({2, 3, 1, 4, 5, 1});
    c0.reshape({3, 2, 1, 2, 10, 1});
    c0.expand({3, 2, 99, 2, 10, 98});

    Chain c1({2, 3, 1, 4, 5, 1});
    c1.expand({2, 3, 99, 4, 5, 98});
    c1.reshape({3, 2, 99, 2, 10, 98});

    c0.canonicalized().confirmEqual(
        c1.canonicalized(),
        "The reshape is localized to be between the expansion dimensions, "
        "expected the expand and reshape to be permutable");
  }

  {
    Chain c0({10, 1, 5});
    c0.reshape({5, 1, 10});
    c0.expand({5, 2, 10});
    auto c1       = c0.canonicalized();
    auto reshapes = c1.where(Type::Reshape);
    auto expands  = c1.where(Type::Expand);
    if (reshapes.size() != 1 || expands.size() != 1 ||
        expands[0] < reshapes[0]) {
      throw poprithms::test::error(
          "The expansion dimension does not localize the reshape. There is "
          "flow across dimension 1.");
    }
  }
}

void testLongerChain0() {

  Chain c0({20, 100, 100});
  c0.slice({0, 0, 0}, {1, 1, 1});
  c0.reshape({});
  c0.expand({1, 100, 100});
  c0.reshape({100, 100});
  c0.slice({0, 0}, {1, 1});
  c0.reshape({});
  c0.expand({100, 100});

  Chain c1({20, 100, 100});
  c1.slice({0, 0, 0}, {1, 1, 1});
  c1.reshape({1, 1});
  c1.expand({100, 100});

  // TODO(T53918) currently not canonicalized to target.
  // c0.canonicalized().confirmEqual(c1.canonicalized());
}

} // namespace

int main() {
  testCanonicalize0();
  testMapToEmpty();
  rubixTwist();
  testBubbleSettSampleReverse0();
  testRedundantSampleFill0();
  testRedundantSampleFill1();
  testExpandDimshuffle0();
  testExpandReverse0();
  testExpandSettSample0();
  testExpandReshape0();
  testLongerChain0();
  return 0;
}
