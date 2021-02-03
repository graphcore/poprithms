// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poprithms/memory/chain/error.hpp>
#include <poprithms/memory/chain/settutil.hpp>
namespace poprithms {
namespace memory {
namespace chain {

void NonNativeSettSampler::assertNonZeroRank(uint64_t inRank) const {
  if (inRank == 0) {
    throw error("Rank 0 input to settSampleFinalDimenion - not permitted.");
  }
}

void NonNativeSettSampler::assertSubCalledRank(const Shape &subCalledShape,
                                               uint64_t inRank) const {
  if (subCalledShape.rank_u64() != inRank + 1) {
    std::ostringstream oss;
    oss << "Expected the Shape of the return "
        << "from recursive call to have rank " << inRank + 1 << ", not "
        << subCalledShape.rank_u64();
    throw error(oss.str());
  }
}

} // namespace chain
} // namespace memory
} // namespace poprithms
