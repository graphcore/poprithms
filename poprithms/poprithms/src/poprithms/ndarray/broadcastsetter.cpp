// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "ndarray/error.hpp"

#include <poprithms/ndarray/broadcastsetter.hpp>

namespace poprithms {
namespace ndarray {

void BroadcastSetter::assertSumeNumElms(uint64_t creation, uint64_t dst) {
  if (creation != dst) {
    std::ostringstream oss;
    oss << "The Tensor created has " << creation << " elements"
        << ", but the target tensor has " << dst
        << ". Something has gone wrong in srcToDst.";
    throw error(oss.str());
  }
}

} // namespace ndarray
} // namespace poprithms
