// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <poprithms/compute/host/regionutil.hpp>
#include <poprithms/compute/host/tensormapper.hpp>

namespace poprithms {
namespace compute {
namespace host {

bool RegionUtil::allZero(const poprithms::compute::host::Tensor &t,
                         const poprithms::memory::nest::Region &r) {
  return TensorMapper::settSample(t, r).allZero();
}

} // namespace host
} // namespace compute
} // namespace poprithms
