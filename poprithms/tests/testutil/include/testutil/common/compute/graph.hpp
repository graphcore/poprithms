// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef TESTUTIL_COMMON_COMPUTE_GRAPH_HPP
#define TESTUTIL_COMMON_COMPUTE_GRAPH_HPP

#include <vector>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/hosttensor.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/ops/withoutcallees.hpp>
#include <poprithms/common/compute/slickgraph.hpp>
#include <poprithms/common/compute/subgraph.hpp>
#include <poprithms/common/compute/tensor.hpp>
#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/op.hpp>

namespace poprithms {
namespace common {
namespace compute {
namespace test {

using Graph =
    poprithms::common::compute::TSlickGraph<Tensor, OptionalTensor, SubGraph>;

} // namespace test
} // namespace compute
} // namespace common
} // namespace poprithms

#endif
