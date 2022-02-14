// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include <common/schedulable/error.hpp>

#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/common/schedulable/graph.hpp>
#include <poprithms/common/schedulable/op.hpp>
#include <poprithms/schedule/vanilla/vanilla.hpp>
#include <poprithms/util/copybyclone_impl.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/stringutil.hpp>
#include <poprithms/util/unisort.hpp>
#include <poprithms/util/where.hpp>

namespace poprithms {
namespace common {
namespace schedulable {

void Graph::multiOutTypeSpecificVerifyValidOutputSubstitute(
    const TensorId &before,
    const TensorId &after) const {

  schedulableTypeSpecificVerifyValidOutputSubstitute(before, after);

  assertSubGraphId({before}, subGraphId(after));
}

// something faster TODO(T42434)
OpIds Graph::vanillaSubSchedule(const std::set<OpId> &opIds) const {

  if (opIds.empty()) {
    return {};
  }

  // If all the Ops are in the same sub-graph SG, we use only the schedule of
  // the sub-graph SG.
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

  if (schedule.size() != opIds.size()) {
    std::ostringstream oss;
    oss << "Failed to obtain a sub-schedule for a set of " << opIds.size()
        << " ops. "
        << "The super-schedule contains " << superSchedule.size()
        << ", but only " << schedule.size()
        << " of the ops in the provided set are in the super-schedule. ";

    throw error(oss.str());
  }
  return schedule;
}

bool Graph::hasUniqueSchedule(SubGraphId sgId) const {
  using namespace schedule;
  return vanilla::Query<uint64_t>::hasUniqueSchedule(
      getForwardEdgeMap_u64(sgId).fwdEdgesCompact(),
      vanilla::VerifyEdges::No);
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

void Graph::propagateControlDependencies(
    OpId opId,
    ControlDependencyPropagationType type) {

  switch (type) {

  case ControlDependencyPropagationType::ConserveLocally: {

    for (auto in_ : controlDependencyInOps(opId)) {
      for (auto o : allOutOps(opId)) {
        constraint(in_, o);
      }
    }

    for (auto out_ : controlDependencyOutOps(opId)) {
      for (auto i : allInOps(opId)) {
        constraint(i, out_);
      }
    }
    break;
  }

  default:
    throw error("Unrecognised ControlDependencyPropagationType");
  }
}

void Graph::SubGraphState::removeOp(OpId opId) {
  ops_.erase(opId);
  if (eagerEnabled()) {
    if (hasKnownLast() && knownLast() == opId) {
      hasLast_ = false;
    }
  }
}

void Graph::multiOutTypeSpecificRemoveOp(
    OpId opId,
    const OptionalTensorIds &substitutes) {

  schedulableTypeSpecificRemoveOp(opId, substitutes);

  for (auto in_ : controlDependencyInOps(opId)) {
    op(in_).removeControlDependencyOut(opId);
  }

  for (auto out_ : controlDependencyOutOps(opId)) {
    op(out_).removeControlDependencyIn(opId);
  }

  auto sgId     = subGraphId(opId);
  auto &sgState = subGraphStates[sgId.get_u64()];
  sgState.removeOp(opId);
}

OpIds Graph::allOutOps(OpId opId) const {
  OpIds outs = dataDependencyOutOps(opId);
  for (auto x : controlDependencyOutOps(opId)) {
    if (std::find(outs.cbegin(), outs.cend(), x) == outs.cend()) {
      outs.push_back(x);
    }
  }
  return outs;
}

OpIds Graph::allInOps(OpId opId) const {
  OpIds ins = dataDependencyInOps(opId);
  for (auto x : controlDependencyInOps(opId)) {
    if (std::find(ins.cbegin(), ins.cend(), x) == ins.cend()) {
      ins.push_back(x);
    }
  }
  return ins;
}

OpIds Graph::dataDependencyInOps(OpId opId) const {
  std::set<OpId> opIds;
  for (auto x : inTensorIds(opId)) {
    opIds.insert(x.opId());
  }
  return OpIds(opIds.cbegin(), opIds.cend());
}

OpIds Graph::dataDependencyOutOps(OpId opId) const {
  std::set<OpId> opIds;
  for (auto x : outTensorIds(opId)) {
    for (auto c : consumptionIds(x)) {
      opIds.insert(c.opId());
    }
  }
  return OpIds(opIds.cbegin(), opIds.cend());
}

OpId Graph::insertSchedulableOp(std::unique_ptr<Op> op_) {

  // insert the new Op into the base multiout::Graph class, which stores all
  // Ops, nut no constraints or subgraph info.
  auto newId = insertMultioutOp(std::move(op_));

  for (auto inId : op(newId).controlDependencyInOps()) {
    constraint(inId, newId);
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

  const auto allOutsOfOp = allOutOps(opId);
  if (!allOutsOfOp.empty()) {
    std::ostringstream oss;
    oss << "Cannot make " << opId
        << " the last of the current ops, at it has "
        << "existing output dependencies: ";
    util::append(oss, allOutsOfOp);
    oss << ". It is impossible for " << opId << " to appear after them. ";
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
    oss << "Cannot insert constraint between Ops in different Graphs. ";
    oss << "Bad constraint, " << op(before) << " -> " << op(after) << ".";
    throw error(oss.str());
  }

  if (before == after) {
    std::ostringstream oss;
    oss << "Cannot insert constraint " << before << " -> " << after
        << ", as this creates a cycle (of size 1). ";
    throw error(oss.str());
  }
  op(before).insertControlDependencyOut(after);
  op(after).insertControlDependencyIn(before);
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

std::vector<uint64_t>
getVanillaSchedule(const std::vector<std::vector<uint64_t>> &fwd) {
  using namespace schedule::vanilla;
  return getSchedule_u64(fwd, ErrorIfCycle::Yes, VerifyEdges::Yes);
}

std::vector<uint64_t>
getRandomSchedule(const std::vector<std::vector<uint64_t>> &fwd,
                  uint32_t seed) {
  using namespace schedule::vanilla;
  return Scheduler<uint64_t, double>::random(
      fwd, {}, {}, seed, ErrorIfCycle::Yes, VerifyEdges::Yes);
}

std::vector<OpIds> Graph::subGraphPartitioned(const OpIds &opIds) const {
  std::vector<OpIds> p(nSubGraphs());
  for (auto id : opIds) {
    p[subGraphId(id).get_u64()].push_back(id);
  }
  return p;
}

OpIds Graph::vanillaSchedule() const {
  const auto fwdEdgeMap = getForwardEdgeMap_u64();
  const auto compactSchedule =
      getVanillaSchedule(fwdEdgeMap.fwdEdgesCompact());

  return fwdEdgeMap.unpacked(compactSchedule);
}

OpIds Graph::randomSchedule(uint32_t s) const {
  const auto fwdEdgeMap = getForwardEdgeMap_u64();
  return fwdEdgeMap.unpacked(
      getRandomSchedule(fwdEdgeMap.fwdEdgesCompact(), s));
}

OpIds Graph::vanillaSchedule(SubGraphId sgId) const {
  const auto fwdEdgeMap = getForwardEdgeMap_u64(sgId);
  const auto compactSchedule =
      getVanillaSchedule(fwdEdgeMap.fwdEdgesCompact());
  return fwdEdgeMap.unpacked(compactSchedule);
}

OpIds Graph::randomSchedule(SubGraphId sgId, uint32_t s) const {
  const auto fwdEdgeMap = getForwardEdgeMap_u64(sgId);
  return fwdEdgeMap.unpacked(
      getRandomSchedule(fwdEdgeMap.fwdEdgesCompact(), s));
}

std::vector<OpIds> Graph::vanillaSchedules() const {
  return subGraphPartitioned(vanillaSchedule());
}

std::vector<OpIds> Graph::randomSchedules(uint32_t s) const {
  return subGraphPartitioned(vanillaSchedule(s));
}

OpIds Graph::controlDependencyInOps(OpId opId) const {
  return op(opId).controlDependencyInOps();
}

OpIds Graph::controlDependencyOutOps(OpId opId) const {
  return op(opId).controlDependencyOutOps();
}

FwdEdgeMap Graph::getSparseForwardEdgeMap_u64(const OpIds &opIds) const {

  FwdEdgeMap fwdEdgeMap(opIds);

  for (auto id : opIds) {
    const auto outs = allOutOps(id);
    fwdEdgeMap.reserve(id, outs.size());
    for (auto out : outs) {
      fwdEdgeMap.insertEdge(id, out);
    }
  }

  return fwdEdgeMap;
}

FwdEdgeMap Graph::getForwardEdgeMap_u64() const {
  return getSparseForwardEdgeMap_u64(common::multiout::Graph::opIds());
}

FwdEdgeMap Graph::getForwardEdgeMap_u64(SubGraphId sgId) const {
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
    if (allOutOps(id).size() == 0) {
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
    const auto nonDataIns = op(i).controlDependencyInOps();
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

  for (auto opId : multiout::Graph::opIds()) {

    for (auto out_ : allOutOps(opId)) {

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

SubGraphIds Graph::asSubGraphIds(const std::vector<uint64_t> &i) const {
  SubGraphIds o;
  o.reserve(i.size());
  for (auto v : i) {
    o.push_back(SubGraphId(v));
  }
  return o;
}

std::vector<uint64_t> Graph::asUnsigned64s(const SubGraphIds &i) const {
  std::vector<uint64_t> o;
  o.reserve(i.size());
  for (auto v : i) {
    o.push_back(v.get_u64());
  }
  return o;
}

} // namespace schedulable
} // namespace common
} // namespace poprithms
