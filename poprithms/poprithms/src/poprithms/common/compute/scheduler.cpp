// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <poprithms/common/compute/scheduler.hpp>

namespace poprithms {
namespace common {
namespace compute {
FwdEdgeMap Scheduler::loweringFwdEdgeMap(const Graph &m) {

  // This initial edge map includes data deps, control deps for
  // non-initialization ops, control deps for initialization ops (which are
  // slid to non-initialization ops, see Op::constraintPhobic), and the deps
  // required to ensure modify inplace ops are final consumers.
  FwdEdgeMap fem = m.getForwardEdgeMap_u64();

  // add edges for inter sub-graph references.
  for (auto opId : m.opIds()) {
    for (uint64_t o = 0; o < m.nOutTensors(opId); ++o) {
      for (auto out : m.computeOp(opId).derivedRefs(o)) {
        fem.insertEdge(opId, out.opId());
      }
    }
  }

  // ops in callees must be lowered before any callers.
  // TODO(T64517) Use bin constraints here (to make faster).
  for (auto opId : m.opIds()) {
    for (auto callee : m.computeOp(opId).callees()) {
      for (auto calleeOp : m.opIds(callee)) {
        fem.insertEdge(calleeOp, opId);
      }
    }
  }

  return fem;
}

OpIds Scheduler::vanillaLoweringSchedule(const Graph &m) {
  const auto fwdEdgeMap = Scheduler::loweringFwdEdgeMap(m);
  using namespace poprithms::schedule::vanilla;
  const auto compactSchedule = getSchedule_u64(
      fwdEdgeMap.fwdEdgesCompact(), ErrorIfCycle::Yes, VerifyEdges::Yes);
  return fwdEdgeMap.unpacked(compactSchedule);
}

SubGraphIds Scheduler::scheduleByRefs(const Graph &m) {

  std::vector<std::vector<uint64_t>> edges;
  edges.resize(m.nSubGraphs());

  for (auto refFromTensor : m.derivedRefs()) {

    const auto refFrom = refFromTensor.opId();
    const auto src = m.subGraphId(m.computeOp(refFrom).rootRef(0)).get_u64();
    const auto dst = m.subGraphId({refFrom, 0}).get_u64();

    if (std::find(edges[src].begin(), edges[src].end(), dst) ==
        edges[src].end()) {
      edges[src].push_back(dst);
    }
  }

  using namespace poprithms::schedule::vanilla;
  auto schedule = getSchedule_u64(edges, ErrorIfCycle::Yes, VerifyEdges::Yes);

  return m.asSubGraphIds(schedule);
}

OpIds Scheduler::vanillaComputeSchedule(const Graph &m, SubGraphId sgId) {
  auto sched = m.vanillaSubGraphSchedule(sgId);
  OpIds nonInitSched;
  for (auto o : sched) {
    if (!m.computeOp(o).isInitializingOp()) {
      nonInitSched.push_back(o);
    }
  }
  return nonInitSched;
}

} // namespace compute
} // namespace common
} // namespace poprithms
