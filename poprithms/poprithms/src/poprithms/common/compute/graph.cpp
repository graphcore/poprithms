// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/schedulable/graph.hpp>

namespace poprithms {
namespace common {
namespace compute {

OpId Graph::insertComputeOp(std::unique_ptr<Op> op) {
  return insertSchedulableOp(std::move(op));
}

DType Graph::dtype(const TensorId &tid) const {
  auto t = computeOp(tid.opId()).outDType(tid.outIndex());
  return t;
}

const Op &Graph::computeOp(OpId a) const {
  // We know that all Ops in this Graph can be safely cast, so no need for
  // dynamic_cast here.
  return static_cast<const Op &>(multioutOp(a));
}

const Op &Graph::op(OpId a) const { return computeOp(a); }

// See Scott Meyers' "Effective C++"
Op &Graph::op(OpId id) {
  return const_cast<Op &>(static_cast<const Graph &>(*this).op(id));
}
} // namespace compute

} // namespace common
} // namespace poprithms
