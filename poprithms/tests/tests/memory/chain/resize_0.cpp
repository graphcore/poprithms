// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <memory/inplace/ops.hpp>
#include <poprithms/memory/chain/chain.hpp>
#include <poprithms/memory/chain/error.hpp>
#include <poprithms/memory/chain/settutil.hpp>
#include <testutil/memory/nest/randomregion.hpp>

namespace {

using namespace poprithms::memory::chain;

// Resize (up-sample by replication)
// from
//       (2,3,5,7)
// to
//       (2,30,5,7).

void test0() {

  // Pre-canonicalization (This approach has unnecessary dimShuffles).
  //
  //  (2,3,5,7) ----> DimShuffle((0,2,3,1))
  //                  Reshape((2,5,7,3,1))
  //                  Expand((2,5,7,3,10))
  //                  Reshape((2,5,7,30))
  //                  DimShuffle((0,3,1,2)) ----> (2,30,5,7)
  //
  //
  // Post-canonicalization
  //  (2,3,5,7) ----> Reshape((2,3,1,5,7))
  //                  Expand((2,3,10,5,7))
  //                  Reshape((2,30,5,7)) ----> (2,30,5,7)

  Shape inShape({2, 3, 5, 7});

  // Chain using the approach of rolling the dimension to the back first:
  Chain ch(inShape);
  const Permutation roller({0, 2, 3, 1});
  ch.dimShuffle(roller);
  const auto s0 = ch.outShape();
  ch.reshape(s0.append(1));
  ch.expand(s0.append(10));
  ch.reshape(s0.scale(Stride(10), Dimension(s0.rank_u64() - 1)));
  ch.dimShuffle(roller.inverse());

  // Approach using the simplified approach
  Chain ch2(inShape);
  const auto unsqueezedShape = inShape.unsqueeze(2);
  ch2.reshape(unsqueezedShape);
  ch2.expand(unsqueezedShape.scale(Stride(10), Dimension(2)));
  ch2.reshape(inShape.scale(Stride(10), Dimension(1)));

  // Confirm that canonicalizing the more complex approach arrives at the
  // simpler approach:
  ch2.confirmEqual(ch.canonicalized());
}

void test1() {

  // (2,3,5) -> dimShuffle(1 2 0) -> reshape(15,1) -> dimShuffle(1 0)
  Shape inShape({2, 3, 5});
  Chain ch(inShape);
  ch.dimShuffle({{1, 2, 0}});
  ch.reshape({30, 1});
  ch.dimShuffle({{1, 0}});

  Chain expected({2, 3, 5});
  expected.dimShuffle({{1, 2, 0}});
  expected.reshape({1, 30});
  ch.canonicalized().confirmEqual(expected);
}

void test2() {

  Chain ch({2, 3, 5});
  ch.dimShuffle({{0, 2, 1}});
  ch.reshape({2, 5, 3, 1});
  ch.expand({2, 5, 3, 7});
  ch.reshape({2, 5, 21});
  ch.dimShuffle({{0, 2, 1}});
  ch.reshape({2, 3, 7, 5});

  auto canon = ch.canonicalized();

  Chain expected({2, 3, 5});
  expected.reshape({2, 3, 1, 5});
  expected.expand({2, 3, 7, 5});
  canon.confirmEqual(expected);
}

void test3() {

  // T35668: make this canonicalization possible.
  //   This is a case where simplification should be possible, but the current
  //   passes don't succeed. The problem seems to be the dimension reducing
  //   Reshapes. If there was another pass which expanded ("canonicalized")
  //   shapes, to be composed of their constituent prime factors, the I thinkf
  //   the low-dimensional dimshuffle could be bubbled backwards.

  Chain ch({2, 3, 5, 7});
  ch.dimShuffle({{1, 2, 3, 0}});
  ch.reshape({3 * 5 * 7 * 2, 1});
  ch.expand({3 * 5 * 7 * 2, 11});
  ch.reshape({3 * 5 * 7, 2 * 11});
  ch.dimShuffle({{1, 0}});
  ch.reshape({2, 11, 3, 5, 7});

  Chain expected({2, 3, 5, 7});
  expected.reshape({2, 1, 3, 5, 7});
  expected.expand({2, 11, 3, 5, 7});

  const auto canonicalized = ch.canonicalized();

  if (canonicalized == expected) {
    std::ostringstream oss;
    oss << "Has Task T35668 been solved?";
    throw error(oss.str());
  }
}

} // namespace

int main() {
  test0();
  test1();
  test2();
  test3();
  return 0;
}
