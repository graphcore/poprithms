// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_PRUNE_PRUNEMUTATOR_HPP
#define POPRITHMS_COMMON_COMPUTE_PRUNE_PRUNEMUTATOR_HPP

#include <map>
#include <memory>
#include <set>
#include <vector>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/ops/withcallees.hpp>
#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/optionaltensorid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/program/callstack/callstack.hpp>
#include <poprithms/program/callstack/copymap.hpp>
#include <poprithms/program/prune/prune.hpp>

namespace poprithms {
namespace common {
namespace compute {

/**
 * Completion of the program::prune::Mutator interface for a compute::Graph.
 * */
class PruneMutator : public poprithms::program::prune::Mutator {

private:
  Graph *graph_;
  Graph &graph() { return *graph_; }

public:
  PruneMutator(Graph &m) : poprithms::program::prune::Mutator(), graph_(&m) {}

  void removeInputs(OpId opId, const InIndices &ins) final {
    graph().removeInputs(opId, ins);
  }

  void removeOutputs(OpId opId, const OutIndices &outs) final {
    graph().removeOutputs(opId, outs, OptionalTensorIds(outs.size()));
  }

  void removeOp(OpId opId, const std::string &ctxt) final;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
