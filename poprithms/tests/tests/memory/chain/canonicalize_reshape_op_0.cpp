// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <sstream>

#include <memory/chain/op.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/chain/chain.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/util/stringutil.hpp>

namespace {

using namespace poprithms::memory::chain;
using namespace poprithms::memory;

bool didSwap(const Shape &inShape,
             const Upper &upper,
             const Shape &outShape) {
  Lower lower{std::vector<int64_t>(inShape.rank_u64())};
  const auto reg = nest::Region::fromBounds(inShape, lower, upper);
  Op x0(Type::SettSample, inShape.slice(lower, upper), reg);
  Op x1(Type::Reshape, outShape, outShape);
  auto swapped = Op::bubbleReshapeBack(inShape, x0, x1);
  return swapped;
}

void assertNotBubblable(const Shape &inShape,
                        const Upper &upper,
                        const Shape &outShape) {
  if (didSwap(inShape, upper, outShape)) {
    throw poprithms::test::error("an impossible bubble with in shape " +
                                 inShape.str() + " and out shape " +
                                 outShape.str());
  }
}

void assertBubblable(const Shape &inShape,
                     const Upper &upper,
                     const Shape &outShape) {
  if (!didSwap(inShape, upper, outShape)) {
    throw poprithms::test::error(
        "a possible bubble with in shape " + inShape.str() +
        " and out shape " + outShape.str() + ", reported as not possible");
  }
}

// Test that the swap does take place, and that the final permuted ops have
// the correct shapes/regions.
void baseTestSettSampleReshape(const Shape &inShape,
                               const Upper &upper,
                               const Shape &outShape,
                               const Shape &interShape) {

  Lower lower{std::vector<int64_t>(inShape.rank_u64())};
  const auto reg = nest::Region::fromBounds(inShape, lower, upper);
  Op x0(Type::SettSample, inShape.slice(lower, upper), reg);
  Op x1(Type::Reshape, outShape, outShape);

  auto getBase = [&]() {
    std::ostringstream oss;
    oss << "Failed in test of bubbling reshape back past sett sample, where "
        << "input shape = " << inShape
        << " and upper bound of slice (from 0) is ";
    poprithms::util::append(oss, upper);
    oss << ". The output shape of the reshape is " << outShape << '.';
    return oss.str();
  };

  auto swapped = Op::bubbleReshapeBack(inShape, x0, x1);
  if (!swapped) {
    throw poprithms::test::error(getBase() + " Failed to swap.");
  }

  if (x0.outShape() != interShape) {
    throw poprithms::test::error(
        getBase() + " Expected new reshape output to be " + interShape.str() +
        " not " + x0.outShape().str());
  }

  auto newLower = Lower{std::vector<int64_t>(interShape.rank_u64(), 0)};

  auto newUpper = Upper{outShape.get()};

  if (!x1.attr().region().equivalent(
          nest::Region::fromBounds(interShape, newLower, newUpper))) {
    std::ostringstream oss;
    oss << getBase() << " Expected new region to be slice with upper bounds ";
    poprithms::util::append(oss, newUpper);
    throw poprithms::test::error(oss.str());
  }
}

void testBubbleSettSampleReshape0() {
  baseTestSettSampleReshape({5, 7, 9}, {2, 2, 9}, {2, 2, 3, 3}, {5, 7, 3, 3});

  baseTestSettSampleReshape(
      {5, 7, 9}, {5, 7, 9}, {5, 1, 7, 1, 3, 3}, {5, 1, 7, 1, 3, 3});

  baseTestSettSampleReshape({}, {}, {}, {});

  baseTestSettSampleReshape({}, {}, {1, 1}, {1, 1});

  baseTestSettSampleReshape(
      {5, 6, 7, 8}, {1, 2, 7, 8}, {1, 2, 4, 2, 7}, {5, 6, 4, 2, 7});

  baseTestSettSampleReshape({8, 9}, {8, 1}, {4, 2, 1}, {4, 2, 9});

  baseTestSettSampleReshape({7, 8, 9}, {7, 8, 1}, {4, 2, 7, 1}, {4, 2, 7, 9});

  baseTestSettSampleReshape(
      {6, 7, 8, 9}, {2, 7, 8, 1}, {2, 4, 2, 7, 1}, {6, 4, 2, 7, 9});

  baseTestSettSampleReshape({5, 6, 7, 8, 9},
                            {1, 2, 7, 8, 1},
                            {1, 2, 4, 2, 7, 1},
                            {5, 6, 4, 2, 7, 9});

  // An ambiguous case: we can only confirm that it's possible, not what the
  // permutation looks like:
  assertBubblable({3}, {1}, {1, 1, 1});

  assertNotBubblable({20, 100, 100}, {1, 1, 1}, {});
  assertNotBubblable({20, 100, 100}, {1, 1, 1}, {1});
  assertNotBubblable({20, 100, 100}, {1, 1, 1}, {1, 1});
  assertNotBubblable({20, 100, 100}, {1, 1, 10}, {1, 10});
  assertNotBubblable({3, 4}, {1, 4}, {2, 1, 2});
  assertNotBubblable({4, 4, 4}, {4, 1, 4}, {8, 1, 2});
  assertNotBubblable({4, 4, 4}, {4, 1, 4}, {2, 1, 8});
  assertNotBubblable({4, 4, 4}, {4, 1, 4}, {2, 2, 2, 1, 2});
}

} // namespace

int main() {
  testBubbleSettSampleReshape0();
  return 0;
}
