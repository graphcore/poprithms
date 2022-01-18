// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "error.hpp"

#include <sstream>

#include <poprithms/common/multiout/fwdedgemap.hpp>
#include <poprithms/memory/unwind/scheduledsolution.hpp>
#include <poprithms/schedule/vanilla/vanilla.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

ScheduledSolution::ScheduledSolution(const Graph &graph,
                                     const Translator &t_,
                                     const FwdEdgeMap &nonUnwindEdgeMap)
    : Solution(graph), em_(nonUnwindEdgeMap) {

  const auto nPaths = barriersToSinks().size();
  const auto nOps   = fwdEdgeMap().nOps();

  // Initialize the node edges map with just the op edge map provided. We will
  // add the paths to the node edge map next.
  auto nodeEdgeMap = fwdEdgeMap().fwdEdgesCompact();

  nodeEdgeMap.resize(nOps + nPaths);

  // The paths to be unwound should be scheduled as late as possible. We
  // ensure this by using a scheduler which uses priorities, and setting the
  // priorities of paths to be negative. This is not strictly necessary, but
  // means there is less redundant work at the poplar level where for example
  // the layout of matmul's output is automatically available if the matmul
  // has already been initialized in the actual compute program, but needs to
  // be 'dummy' computed otherwise.
  constexpr double aNegVal{-1.};
  std::vector<std::tuple<uint64_t, double>> priorities;
  for (uint64_t pId = 0; pId < nPaths; ++pId) {
    const NodeId pathNodeId{nOps + pId};
    priorities.push_back(
        std::tuple<uint64_t, double>{pathNodeId.get(), aNegVal});
    const auto &p         = barriersToSinks()[pId];
    const auto machineDst = t_.fromUnwind(p.dst()).opId();
    const auto nodeDst    = fwdEdgeMap().compactId(machineDst);
    nodeEdgeMap[pathNodeId.get()].push_back(nodeDst);

    // It is strictly required that paths are unwound in the order provided by
    // the poprithms solution. Consider for example:
    //
    // a <- init({10});
    // b <- a.slice_({0}, {9}).reshape_({3,3});
    // c <- a.slice_({9}, {10})
    // d <- b + c.
    // e <- matmul(b,b).
    //
    // The layout of b (from matmul) is required before the layout of c can be
    // determined (broadcast operand). Switching the order of path unwinding
    // is not allowed!
    //
    if (pId > 0) {
      nodeEdgeMap[pathNodeId.get() - 1].push_back(pathNodeId.get());
    }
  }

  for (auto i :
       poprithms::schedule::vanilla::Scheduler<uint64_t, double>::filo(
           nodeEdgeMap,
           priorities,
           {},
           poprithms::schedule::vanilla::ErrorIfCycle::Yes,
           poprithms::schedule::vanilla::VerifyEdges::Yes)) {
    schedule_.push_back(NodeId(i));
  }

  summary_ = createSummary(t_);
}

bool ScheduledSolution::isPathToSink(NodeId id) const {
  return id.get() >= opsEnd() && id.get() < pathsEnd();
}

std::string ScheduledSolution::createSummary(const Translator &t) const {
  std::ostringstream ost;
  ost << "ScheduledSolution order :";
  std::string spc_{"        "};
  for (auto n : schedule_) {
    ost << '\n' << spc_;
    if (n.get() < opsEnd()) {
      auto opId = op(n);
      ost << "Op : " << t.str(opId);
    } else {
      const auto &up = pathToSink(n);
      ost << "Path to " << t.fromUnwind(up.dst()) << " : " << up;
    }
  }
  return ost.str();
}

bool ScheduledSolution::isOp(NodeId id) const { return id.get() < opsEnd(); }

const Path &ScheduledSolution::pathToSink(NodeId id) const {

  if (isPathToSink(id)) {
    return barriersToSinks()[id.get() - opsEnd()];
  }

  std::ostringstream oss;
  oss << "Invalid NodeId (id=" << id << ") in ScheduledSolution::pathToSink. "
      << "All NodeIds corresponding "
      << "to Paths are in the range [" << opsEnd() << ", " << pathsEnd()
      << "). ";
  throw error(oss.str());
}

OpId ScheduledSolution::op(NodeId id) const {
  if (isOp(id)) {
    return fwdEdgeMap().opId(id.get());
  }

  std::ostringstream oss;
  oss << "Invalid NodeId (id=" << id << ") in ScheduledSolution::op. "
      << "All NodeIds corresponding "
      << "to Ops are in the range [0, " << opsEnd() << "). ";
  throw error(oss.str());
}

} // namespace unwind
} // namespace memory
} // namespace poprithms
