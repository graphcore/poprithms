// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_UNWIND_SCHEDULEDSOLUTION_HPP
#define POPRITHMS_MEMORY_UNWIND_SCHEDULEDSOLUTION_HPP

#include <poprithms/common/multiout/fwdedgemap.hpp>
#include <poprithms/memory/unwind/solution.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

using NodeId  = poprithms::util::TypedInteger<'d', uint64_t>;
using NodeIds = std::vector<NodeId>;

/**
 * Abstract base class, which has 2 pure virtual methods (described below).
 * */
class Translator {
public:
  /**
   * Given a sink tensor in an unwind graph, provide the corresponding
   * 'input' tensor in the external graph, whose layout we want to set.
   * */
  virtual TensorId fromUnwind(const TensorId &uwId) const = 0;

  /**
   * Provide debug information for an op in the external graph. This
   * method is only used for logging purposes.
   * */
  virtual std::string str(OpId xtId) const = 0;
};

/**
 * This class creates a schedule from a DAG of nodes, where each node
 * corresponds either to
 *
 * (1) an op in the external graph which a memory::unwind graph
 *     models, or
 * (2) a path from a barrier to a sink in the memory::unwind graph.
 *
 * Recall that each sink in the memory::unwind graph corresponds to an
 * initialization op in the external graph. For each such sink, there might be
 * several paths leading to it from barriers. The nodes corresponding to these
 * paths are all scheduled before the node corresponding to the
 * initialization of the op itself.
 *
 * From a poplar perspective, this class ensures that all variables have
 * complete tile mappings before they are used. The complete tile mappings are
 * obtained by unwinding along one or several paths, in a valid order.
 * */
class ScheduledSolution : public Solution {

  using FwdEdgeMap = poprithms::common::multiout::FwdEdgeMap;

public:
  /**
   * \param graph A memory::unwind graph, complete with sinks, barriers, etc.
   *
   * \param translator An object which provides a mapping between all sink
   *                   tensors in the unwind graph #graph, and the
   *                   corresponding initialization ops in the external
   *                   graph, which the unwind graph corresponds to.
   *
   * \param xtEdgeMap The constraints between ops in the external graph
   *                  on their respective lowering order. These might be data
   *                  dependencies, control dependencies inserted to obtain a
   *                  good liveness profile, dependencies imposed by
   *                  sub-graphs where all of the ops in an op's callee must
   *                  be lowered before the caller, or anything else.
   */

  explicit ScheduledSolution(const Graph &graph,
                             const Translator &translator,
                             const FwdEdgeMap &xtEdgeMap);

  /**
   * The total number of nodes: the number of ops in the external graph, plus
   * the number of paths from barriers to sinks which must be unwound to
   * obtain complete coverage of the input tensors.
   * */
  uint64_t nNodes() const { return schedule_.size(); }

  /**
   * \return true if the node #nid corresponds to an unwind path.
   * */
  bool isPathToSink(NodeId nid) const;
  const Path &pathToSink(NodeId) const;

  /**
   * \return true if the node #nid corresponds to a op in the external graph.
   * */
  bool isOp(NodeId) const;
  OpId op(NodeId) const;

  const std::string &summary() const { return summary_; }

  const NodeIds &schedule() const { return schedule_; }

  NodeId schedule(uint64_t i) const { return schedule_[i]; }

private:
  std::string createSummary(const Translator &t) const;

  const FwdEdgeMap &fwdEdgeMap() const { return em_; }
  NodeIds schedule_;
  FwdEdgeMap em_;
  std::string summary_;

  uint64_t opsEnd() const { return fwdEdgeMap().nOps(); }

  uint64_t pathsEnd() const { return opsEnd() + barriersToSinks().size(); }
};

std::ostream &operator<<(std::ostream &, const ScheduledSolution &);

} // namespace unwind
} // namespace memory
} // namespace poprithms

#endif
