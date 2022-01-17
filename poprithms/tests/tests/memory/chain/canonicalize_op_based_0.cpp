// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <sstream>

#include <memory/chain/op.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/chain/chain.hpp>

namespace {

using namespace poprithms::memory::chain;

void testBubbleDimShuffleReverse0() {

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
    throw poprithms::test::error("Failed to swap reverse and dimShuffle");
  }
  if (x0.type() != Type::Reverse) {
    throw poprithms::test::error(
        "x0 and x1 should have had their types swapped");
  }
  if (x0.attr().dimensions() != Dimensions({1})) {
    std::ostringstream oss;
    oss << "Before the swap, dimension 0 was reversed after the permutation"
        << "(1 2 0)"
        << ". dimension 0 after the permutation corresponds to dimension 1"
        << " before the permutation. Therefore expected the Dimensions of "
        << "the Reverse before the DimShuffle to be {1}.";
    throw poprithms::test::error(oss.str());
  }
}

void testBubbleSettFillIntoExpand0() {

  using namespace poprithms::memory;
  std::vector<nest::Sett> setts{
      poprithms::memory::nest::Sett{{{7, 3, 1}}},
      poprithms::memory::nest::Sett::createAlwaysOn()};

  Region rFill0({10, 1}, setts);

  Chain c({7, 1});
  c.settFillInto(rFill0);
  c.expand({10, 4});

  auto baseMessage = [&c]() {
    std::ostringstream oss;
    oss << "Error testing (SettFillInto, Expand) permuting, for Chain \n"
        << c << "\n";
    return oss.str();
  };

  auto canon = c.canonicalized();

  Op x0(Type::SettFillInto, {10, 1}, rFill0);
  Op x1(Type::Expand, {10, 4}, Shape{10, 4});
  auto swapped = Op::bubbleExpandBack({7, 1}, x0, x1);
  if (!swapped) {
    throw poprithms::test::error(baseMessage() +
                                 "This (SettFillInto, Expand) is permutable");
  }

  Op x2(Type::Expand, {7, 4}, Shape({7, 4}));
  Op x3(Type::SettFillInto, {10, 4}, Region({10, 4}, setts));

  if (x1 != x3 || x0 != x2) {
    throw poprithms::test::error(
        baseMessage() +
        "Unexpected result of permuting the SettFillInto and the Expand");
  }
}
} // namespace

int main() {
  testBubbleDimShuffleReverse0();
  testBubbleSettFillIntoExpand0();
  return 0;
}
