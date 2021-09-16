// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <common/schedulable/error.hpp>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include <poprithms/common/schedulable/graph.hpp>
#include <poprithms/common/schedulable/op.hpp>
#include <poprithms/schedule/shift/graph.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>
#include <poprithms/schedule/vanilla/vanilla.hpp>
#include <poprithms/util/copybyclone_impl.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/stringutil.hpp>
#include <poprithms/util/unisort.hpp>
#include <poprithms/util/where.hpp>

namespace poprithms {
namespace common {
namespace schedulable {

void Graph::removeSchedulableOp(OpId opId,
                                const OptionalTensorIds &substitutes,
                                const std::string &context) {

  assertSubGraphId(util::nonOptionals<TensorId>(substitutes),
                   subGraphId(opId));

  // transfer constraints. Note that we've lost track of which constraints are
  // 'data' and 'non-data', so there might be some constraints in here which
  // are no longer desired. These must be removed separately.
  binConstraint({inOps(opId), outOps(opId)});
  for (auto in_ : inOps(opId)) {
    op(in_).removeOut(opId);
  }
  for (auto out_ : outOps(opId)) {
    op(out_).removeIn(opId);
  }

  removeMultioutOp(opId, substitutes, context);
}

// something faster TODO(T42434)
OpIds Graph::vanillaSubSchedule(const std::set<OpId> &opIds) const {

  if (opIds.empty()) {
    return {};
  }

  // If all the Ops in the same sub-graph, we use only the schedule of that
  // sub-graph.
  const auto superSchedule = [this, &opIds]() {
    const auto sg0 = subGraphId(*opIds.cbegin());
    if (std::all_of(opIds.cbegin(), opIds.cend(), [this, sg0](auto opId) {
          return subGraphId(opId) == sg0;
        })) {
      return vanillaSchedule(sg0);
    }
    return vanillaSchedule();
  }();

  // sort by topological order in fwd graph.
  std::vector<OpId> schedule;
  schedule.reserve(opIds.size());
  for (auto opId : superSchedule) {
    if (opIds.count(opId) != 0) {
      schedule.push_back(opId);
    }
  }
  return schedule;
}

bool Graph::hasUniqueSchedule(SubGraphId sgId) const {
  using namespace schedule;
  return vanilla::Query<uint64_t>::hasUniqueSchedule(
      getForwardEdgeMap_u64(sgId).fwdEdgesCompact, vanilla::VerifyEdges::No);
  return true;
}

void Graph::assertSubGraphId(const TensorIds &tIds,
                             SubGraphId subGraphId) const {
  if (!tIds.empty() && subGraphIdFromTensorIds(tIds) != subGraphId) {
    std::ostringstream oss;
    oss << "\nThe SubGraphIds of the Tensors " << tIds << " are "
        << subGraphIds(tIds)
        << ", but the target SubGraphId in assertSubGraphId is " << subGraphId
        << ": Invalid SubGraphId in assertion, ";
    throw error(oss.str());
  }
}

SubGraphId
Graph::subGraphIdFromTensorIds(const std::vector<TensorIds> &tidss) const {
  uint64_t n = std::accumulate(
      tidss.cbegin(), tidss.cend(), 0ULL, [](uint64_t nIn, const auto &x) {
        return nIn + x.size();
      });

  TensorIds flat{};
  flat.reserve(n);

  for (const auto &x : tidss) {
    flat.insert(flat.end(), x.cbegin(), x.cend());
  }
  return subGraphIdFromTensorIds(flat);
}

SubGraphId Graph::subGraphIdFromTensorIds(const TensorIds &ids) const {
  if (ids.empty()) {
    throw error(
        "Failed to obtain SubGraphId from empty vector of TensorIds. ");
  }

  const auto subGraphId_ = subGraphId(ids[0]);
  if (std::any_of(
          ids.cbegin() + 1, ids.cend(), [&subGraphId_, this](const auto &id) {
            return subGraphId(id) != subGraphId_;
          })) {
    std::ostringstream oss;
    oss << "Contradictory solution while attemting to obtain "
        << "SubGraphId from the TensorIds, " << ids
        << ". Expected all TensorIds to have same SubGraphId, "
        << "but the SubGraphIds are not all identical, " << subGraphIds(ids);
    throw error(oss.str());
  }

  return subGraphId_;
}

void Graph::binConstraint(const std::vector<OpIds> &bins) {
  std::vector<uint64_t> nonEmpty;

  auto sgId = SubGraphId::unset();
  for (const auto &opIds : bins) {
    for (auto opId : opIds) {
      if (sgId == SubGraphId::unset()) {
        sgId = subGraphId(opId);
      }
      if (sgId != subGraphId(opId)) {
        throw error("All Ops in all bins must be have the same SubGraphId");
      }
    }
  }

  for (uint64_t i = 0; i < bins.size(); ++i) {
    if (!bins[i].empty()) {
      nonEmpty.push_back(i);
    }
  }

  if (nonEmpty.size() > 1) {
    for (uint64_t i = 1; i < nonEmpty.size(); ++i) {
      // single op to manage all nIn * nOut constraints.
      auto boundary = insertBinBoundary(sgId);
      for (auto opId : bins[nonEmpty[i - 1]]) {
        constraint(opId, boundary);
      }
      for (auto opId : bins[nonEmpty[i]]) {
        constraint(boundary, opId);
      }
    }
  }
}

const Op &Graph::op(OpId a) const {
  // We know that all Ops in this Graph can be safely cast to class
  // schedulable::Op, as there is no mechanism for an Op which does not
  // inherit from schedulable::Op to enter this Graph.
  return static_cast<const Op &>(multioutOp(a));
}

// See Scott Meyers' "Effective C++"
Op &Graph::op(OpId id) {
  return const_cast<Op &>(static_cast<const Graph &>(*this).op(id));
}

bool Graph::eagerIsEnabled(SubGraphId id_) const {
  return subGraphStates[id_.get_u64()].eagerEnabled();
}

OpId Graph::insertSchedulableOp(std::unique_ptr<Op> op_) {

  // insert the new Op into the base multiout::Graph class, which stores all
  // Ops, nut no constraints or subgraph info.
  auto newId = insertMultioutOp(std::move(op_));

  for (auto inId : op(newId).inTensorIds()) {
    constraint(inId.opId(), newId);
  }

  // subgraph specific registering:
  auto &info = subGraphStates[subGraphId(newId).get_u64()];
  info.insertBack(newId);

  if (info.eagerEnabled()) {

    // If this isn't the first insertion in Eager mode, ensure it's after the
    // last one added in eager mode.
    if (info.hasKnownLast()) {
      constraint(info.knownLast(), newId);
    }

    info.setLast(newId);
  }

  return newId;
}

void Graph::ensureLastOfCurrentOps(OpId opId) {

  if (!outOps(opId).empty()) {
    std::ostringstream oss;
    oss << "Cannot make " << opId
        << " the last of the current ops, at it has "
        << "existing output dependencies: ";
    util::append(oss, outOps(opId));
    throw error(oss.str());
  }

  for (auto previous : mayBeFinals(subGraphId(opId))) {
    if (previous != opId) {
      constraint(previous, opId);
    }
  }
}

void Graph::toggleEager(SubGraphId subGraphId, bool enabled) {
  subGraphStates[subGraphId.get_u64()].toggleEager(enabled);
}

void Graph::SubGraphState::toggleEager(bool enabled) {
  if (enabled != eagerEnabled()) {
    eager_   = enabled ? Eager::Enabled : Eager::Disabled;
    hasLast_ = false;
    last_    = -1;
  }
}

void Graph::constraint(OpId before, OpId after) {
  if (op(before).subGraphId() != op(after).subGraphId()) {
    std::ostringstream oss;
    oss << "Cannot insert Constraint between Ops in different Graphs. ";
    oss << "Bad constraint, " << op(before) << " -> " << op(after) << ".";
    throw error(oss.str());
  }
  op(before).insertOut(after);
  op(after).insertIn(before);
}

void Graph::constraint(OpId a, const OpIds &bs) {
  for (auto b : bs) {
    constraint(a, b);
  }
}

void Graph::constraint(const OpIds &as, OpId b) {
  for (auto a : as) {
    constraint(a, b);
  }
}

SubGraphId Graph::createSubGraphId(const std::string &n) {
  subGraphStates.push_back({n});
  return subGraphStates.size() - 1;
}

std::string Graph::subGraphName(SubGraphId id) const {
  return subGraphStates[id.get_u64()].name();
}

std::ostream &operator<<(std::ostream &ost, const Graph::FwdEdgeMap &fem) {
  auto fromCompact     = fem.fromCompact;
  auto fwdEdgesCompact = fem.fwdEdgesCompact;
  ost << "from compact" << '\n';
  ost << "------------" << '\n';
  for (uint64_t i = 0; i < fromCompact.size(); ++i) {
    ost << ' ' << i << " --> " << fromCompact[i] << '\n';
  }

  ost << "Compact edges" << '\n';
  ost << "-------------" << '\n';
  for (uint64_t i = 0; i < fwdEdgesCompact.size(); ++i) {
    ost << ' ' << i << " --> ";
    util::append(ost, fwdEdgesCompact[i]);
    ost << '\n';
  }
  return ost;
}

namespace {

// Convert a compact schedule into a non-compact (might have holes) one.
OpIds unpacked(const Graph::FwdEdgeMap &fwdEdgeMap,
               const std::vector<uint64_t> &s_u64) {
  OpIds f;
  f.reserve(s_u64.size());
  for (auto v : s_u64) {
    f.push_back(fwdEdgeMap.fromCompact[v]);
  }
  return f;
}

std::vector<uint64_t>
getVanillaSchedule(const std::vector<std::vector<uint64_t>> &fwd) {
  using namespace schedule::vanilla;
  return getSchedule_u64(fwd, ErrorIfCycle::Yes, VerifyEdges::Yes);
}

std::vector<uint64_t>
getRandomSchedule(const std::vector<std::vector<uint64_t>> &fwd,
                  uint32_t seed) {
  using namespace schedule::shift;
  auto g    = schedule::shift::Graph(fwd);
  auto soln = schedule::shift::ScheduledGraph(
      std::move(g),
      KahnTieBreaker::RANDOM,
      TransitiveClosureOptimizations::allOff(),
      RotationTermination::preStart(),
      RotationAlgo::RIPPLE,
      seed,
      FileWriter::None(),
      DebugMode::Off);

  std::vector<uint64_t> schedule(fwd.size());
  for (uint64_t i = 0; i < fwd.size(); ++i) {
    schedule[i] = soln.scheduleToOp(static_cast<int64_t>(i));
  }
  return schedule;
}

} // namespace

std::vector<OpIds> Graph::subGraphPartitioned(const OpIds &opIds) const {
  std::vector<OpIds> p(nSubGraphs());
  for (auto id : opIds) {
    p[subGraphId(id).get_u64()].push_back(id);
  }
  return p;
}

OpIds Graph::vanillaSchedule() const {
  const auto fwdEdgeMap = getForwardEdgeMap_u64();
  return unpacked(fwdEdgeMap, getVanillaSchedule(fwdEdgeMap.fwdEdgesCompact));
}

OpIds Graph::randomSchedule(uint32_t s) const {
  const auto fwdEdgeMap = getForwardEdgeMap_u64();
  return unpacked(fwdEdgeMap,
                  getRandomSchedule(fwdEdgeMap.fwdEdgesCompact, s));
}

OpIds Graph::vanillaSchedule(SubGraphId sgId) const {
  const auto fwdEdgeMap = getForwardEdgeMap_u64(sgId);
  return unpacked(fwdEdgeMap, getVanillaSchedule(fwdEdgeMap.fwdEdgesCompact));
}

OpIds Graph::randomSchedule(SubGraphId sgId, uint32_t s) const {
  const auto fwdEdgeMap = getForwardEdgeMap_u64(sgId);
  return unpacked(fwdEdgeMap,
                  getRandomSchedule(fwdEdgeMap.fwdEdgesCompact, s));
}

std::vector<OpIds> Graph::vanillaSchedules() const {
  return subGraphPartitioned(vanillaSchedule());
}

std::vector<OpIds> Graph::randomSchedules(uint32_t s) const {
  return subGraphPartitioned(vanillaSchedule(s));
}

OpIds Graph::inOps(OpId opId) const { return op(opId).inOps(); }

OpIds Graph::outOps(OpId opId) const { return op(opId).outOps(); }

Graph::FwdEdgeMap
Graph::getSparseForwardEdgeMap_u64(const OpIds &opIds) const {

  std::vector<std::vector<uint64_t>> fEdges(opIds.size());
  OpIds fromCompact;

  std::unordered_map<OpId, uint64_t> toCompact;
  for (auto id : opIds) {
    toCompact.insert(/*hint = */ toCompact.end(), {id, fromCompact.size()});
    fromCompact.push_back(id);
  }

  for (auto id : opIds) {
    const auto outs      = outOps(id);
    const auto compactId = toCompact[id];
    fEdges[compactId].reserve(outs.size());
    for (auto out : outs) {
      fEdges[compactId].push_back(toCompact[out]);
    }
  }

  return {fEdges, fromCompact};
}

Graph::FwdEdgeMap Graph::getForwardEdgeMap_u64() const {
  return getSparseForwardEdgeMap_u64(common::multiout::Graph::opIds());
}

Graph::FwdEdgeMap Graph::getForwardEdgeMap_u64(SubGraphId sgId) const {
  return getSparseForwardEdgeMap_u64(opIds(sgId));
}

SubGraphIds Graph::subGraphIds(const TensorIds &ids) const {
  SubGraphIds subGraphIds;
  subGraphIds.reserve(ids.size());
  for (const auto &id : ids) {
    subGraphIds.push_back(subGraphId(id));
  }
  return subGraphIds;
}

// We could consider keeping an additional datastructre (a field in
// SubGraphState) which partitions OpIds by Graph, but in my (PopART)
// experience it's often that this partition is needed.
OpIds Graph::opIds(SubGraphId subGraphId_) const {
  return subGraphStates.at(subGraphId_.get_u64()).ops();
}

SubGraphId Graph::subGraphId(const TensorId &id) const {
  return subGraphId(id.opId());
}

SubGraphId Graph::subGraphId(OpId i) const { return op(i).subGraphId(); }

OpIds Graph::mayBeFinals(SubGraphId subGraphId) const {
  OpIds tailEnders;
  const auto ids = opIds(subGraphId);
  for (auto id : ids) {
    if (op(id).outOps().empty()) {
      tailEnders.push_back(id);
    }
  }
  return tailEnders;
}

void Graph::link(OpId before, OpId after) {
  std::ostringstream oss;
  oss << "Call to schedulable::Graph::link(before = " << before
      << ", after = " << after
      << "). This method has not been implemented yet. ";
  throw error(oss.str());
}

void Graph::link(const OpIds &opIds) {
  if (opIds.size() > 1) {
    for (uint64_t i = 1; i < opIds.size(); ++i) {
      link(opIds[i - 1], opIds[i]);
    }
  }
}

void Graph::simplifyLinks() {
  throw error("Call to schedulable::Graph::simplifyLinks. This method has "
              "not yet been implemented");
}

std::vector<poprithms::util::StringColumn>
Graph::getSchedulableColumns(const OpIds &opIds) const {

  std::vector<poprithms::util::StringColumn> cols;

  const auto nTens = nMultioutRows(opIds);
  using Strings    = std::vector<std::string>;
  // extensions:
  Strings sg__(nTens, "");

  Strings nonDataIns__(nTens, "");
  bool hasNonDataIns{false};

  uint64_t ti{0};
  for (auto i : opIds) {
    const auto nonDataIns = op(i).nonDataInOps();
    hasNonDataIns |= !nonDataIns.empty();
    nonDataIns__[ti]      = util::getStr(nonDataIns);
    const auto subGraphId = op(i).subGraphId();
    const auto gName      = subGraphName(subGraphId);
    sg__[ti]              = gName.empty() ? subGraphId.str()
                             : gName + "(id=" + subGraphId.str() + ")";
    for (uint64_t o = 0; o < op(i).nOutTensors(); ++o) {
      ++ti;
    }
    if (op(i).nOutTensors() == 0) {
      ++ti;
    }
  }

  cols.push_back({"Graph", std::move(sg__)});
  if (hasNonDataIns) {
    cols.push_back({"NonDataIns", std::move(nonDataIns__)});
  }
  return cols;
}

TensorIds Graph::tensorIds(SubGraphId subGraphId) const {
  TensorIds tensorIds;
  for (auto opId : opIds(subGraphId)) {
    for (auto outId : outTensorIds(opId)) {
      tensorIds.push_back(outId);
    }
  }
  return tensorIds;
}

void Graph::assertSchedulableGraphCorrectness() const {

  // check the base class is in a correct state:
  assertMultioutGraphCorrectness();

  // check that sub-graphs in in/out relationships agree.
  for (auto opId : multiout::Graph::opIds()) {
    for (auto in_ : inOps(opId)) {
      if (!op(in_).isOut(opId)) {
        std::ostringstream oss;
        oss << "Op " << opId << " has " << in_ << " as an in dep, but " << in_
            << " doesn't have " << opId
            << " as an out dep. Correctness assertion failed. ";
        throw error(oss.str());
      }
    }

    for (auto out_ : outOps(opId)) {
      if (!op(out_).isIn(opId)) {
        std::ostringstream oss;
        oss << "Op " << opId << " has " << out_ << " as an out dep, but "
            << out_ << " doesn't have " << opId
            << " as an in dep. Correctness assertion failed. ";
        throw error(oss.str());
      }

      if (subGraphId(out_) != subGraphId(opId)) {
        std::ostringstream oss;
        oss << "There is a topological constraint, " << opId << " -> " << out_
            << ", but these 2 ops are not in the same sub graph. "
            << " Correctness assertion failed. ";
        throw error(oss.str());
      }
    }
  }
}

} // namespace schedulable
} // namespace common
} // namespace poprithms
