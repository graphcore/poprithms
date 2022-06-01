// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef TESTUTIL_COMMON_COMPUTE_GRAPH_HPP
#define TESTUTIL_COMMON_COMPUTE_GRAPH_HPP

#include <vector>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/hosttensor.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/ops/withoutcallees.hpp>
#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/op.hpp>

namespace poprithms {
namespace common {
namespace compute {
namespace test {

/**
 * A minimal completion of the abstract compute::Graph class.
 * */
class Graph : public poprithms::common::compute::Graph {

private:
  [[noreturn]] void noImpl() const {
    throw poprithms::test::error("not implemented");
  }

public:
  Graph() = default;
  Graph(uint64_t nTilesPerReplica, compute::ReplicationFactor rf)
      : poprithms::common::compute::Graph(nTilesPerReplica, rf) {}

  bool multiOutTypeSpecificEqualTo(
      const poprithms::common::multiout::Graph &rhs) const final;

  OpId insertBinBoundary(SubGraphId) final {
    unimplemented("insertBinBoundary");
  }

  std::map<OpId, OpIds>
  schedulableDerivedSpecificConstraints(const OpIds &) const final {
    unimplemented("schedulableDerivedSpecificConstraints");
  }
};

} // namespace test
} // namespace compute
} // namespace common
} // namespace poprithms

#endif
