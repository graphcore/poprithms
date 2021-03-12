// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <memory/chain/op.hpp>
#include <memory/inplace/ops.hpp>
#include <poprithms/memory/chain/chain.hpp>
#include <poprithms/memory/chain/error.hpp>
#include <poprithms/memory/chain/settutil.hpp>
#include <testutil/memory/nest/randomregion.hpp>

namespace {
using namespace poprithms::memory::chain;

void testBubbleReverseDimShuffle0() {

  //  (2,3,5,7) ----> Reverse((0))
  //                  DimShuffle((1,2,3,0))  ----> (3,5,7,2)
  // becomes:
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

void testBubbleReverseDimShuffle1() {

  //  (2,3,5) ----> Reverse((0,1))
  //                DimShuffle((1 2 0))  ----> (3,5,2)
  // becomes:
  //
  //  (2,3,5) ----> DimShuffle((1 2 0))
  //                Reverse((0,2))       ----> (3,5,2)

  Chain c({2, 3, 5});
  c.reverse(Dimensions({0, 1}));
  c.dimShuffle({{1, 2, 0}});
  c.canonicalize();

  Chain expected({2, 3, 5});
  expected.dimShuffle({{1, 2, 0}});
  expected.reverse(Dimensions({0, 2}));
  expected.confirmEqual(c);
}

void testBubbleSettSampleDimShuffle0() {
  Chain c({20, 30, 50});
  c.slice({0, 0, 0}, {15, 25, 45});
  c.dimShuffle({{1, 2, 0}});
  c.canonicalize();

  Chain expected({20, 30, 50});
  expected.dimShuffle({{1, 2, 0}});
  expected.slice({0, 0, 0}, {25, 45, 15});
  expected.confirmEqual(c);
}

} // namespace

int main() {

  testBubbleSettSampleDimShuffle0();
  testBubbleReverseDimShuffle0();
  testBubbleReverseDimShuffle1();

  return 0;
}
