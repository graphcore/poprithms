// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <testutil/common/compute/graph.hpp>

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace common {
namespace compute {
namespace test {

bool Graph::multiOutTypeSpecificEqualTo(
    const poprithms::common::multiout::Graph &rhs) const {
  const auto r                  = static_cast<const Graph &>(rhs);
  const auto atComputeLevel     = computeTypeSpecificEqualTo(r);
  const auto atSchedulabelLevel = schedulableTypeSpecificEqualTo(r);
  return atSchedulabelLevel && atComputeLevel;
}
} // namespace test
} // namespace compute
} // namespace common
} // namespace poprithms
