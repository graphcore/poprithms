// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

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
  a.canonicalize().confirmEqual(Chain({4, 5, 6, 7}));
}

} // namespace

int main() {
  testCanonicalize0();
  return 0;
}
