// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "poprithms/error/error.hpp"

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

#include <common/schedulable/bidiredgemap.hpp>
#include <common/schedulable/error.hpp>

#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/common/schedulable/graph.hpp>
#include <poprithms/common/schedulable/op.hpp>
#include <poprithms/schedule/scc/scc.hpp>
#include <poprithms/schedule/vanilla/vanilla.hpp>
#include <poprithms/util/copybyclone_impl.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/stringutil.hpp>
#include <poprithms/util/unisort.hpp>
#include <poprithms/util/where.hpp>

namespace poprithms {
namespace common {
namespace schedulable {

std::vector<poprithms::util::StringColumn> Graph::getSchedulableColumns(
    const poprithms::util::StringColumn::Parameters &p) const {
  return getSchedulableColumns(multiout::Graph::opIds(), p);
}

void Graph::multiOutTypeSpecificVerifyValidSubstitute(
    const TensorId &before,
    const TensorId &after) const {

  schedulableTypeSpecificVerifyValidSubstitute(before, after);

  verifySubGraphId({before}, subGraphId(after));
}

// something faster TODO(T42434)
OpIds Graph::vanillaSubSchedule(const std::set<OpId> &opIds,
                                const AdditionalFwdEdges &ae) const {

  if (opIds.empty()) {
    return {};
  }

  // If all the Ops are in the same sub-graph SG, we use only the schedule of
  // the sub-graph SG.
  const auto superSchedule = [this, &opIds, &ae]() {
    const auto sg0 = subGraphId(*opIds.cbegin());
    if (std::all_of(opIds.cbegin(), opIds.cend(), [this, sg0](auto opId) {
          return subGraphId(opId) == sg0;
        })) {
      return vanillaSubGraphSchedule(sg0, ae);
    }
    return vanillaSchedule(ae);
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
        << " ops. The super-schedule contains " << superSchedule.size()
        << ", but only " << schedule.size()
        << " of the ops in the provided set are in the super-schedule. ";

    throw error(oss.str());
  }
  return schedule;
}

bool Graph::hasUniqueSchedule(SubGraphId sgId,
                              const AdditionalFwdEdges &ae) const {
  using namespace schedule;
  return vanilla::Query<uint64_t>::hasUniqueSchedule(
      getSubGraphForwardEdgeMap_u64(sgId, ae).fwdEdgesCompact(),
      vanilla::VerifyEdges::No);
  return true;
}

bool Graph::isSchedulable(const AdditionalFwdEdges &afe) const {

  using namespace schedule;
  return vanilla::Query<uint64_t>::isSchedulable(
      getForwardEdgeMap_u64(afe).fwdEdgesCompact(), vanilla::VerifyEdges::No);
}

void Graph::verifySubGraphId(const TensorIds &tIds,
                             SubGraphId subGraphId) const {
  if (!tIds.empty() && subGraphIdFromTensorIds(tIds) != subGraphId) {
    std::ostringstream oss;
    oss << "\nThe SubGraphIds of the Tensors " << tIds << " are "
        << subGraphIds(tIds)
        << ", but the target SubGraphId in verifySubGraphId is " << subGraphId
        << ": Invalid SubGraphId in verifyion, ";
    throw error(oss.str());
  }
}

SubGraphId
Graph::subGraphIdFromTensorIds(const std::vector<TensorIds> &tidss) const {
  return subGraphIdFromTensorIds(TensorId::flatten(tidss));
}

SubGraphId Graph::subGraphIdFromTensorIds(const TensorIds &ids) const {
  return SubGraphId::fromTensorIds(*this, ids);
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
    } else {
      std::ostringstream oss;
      oss << "Error in allOutOps for " << op(opId)
          << ", where data dependency outs are ";
      util::append(oss, dataDependencyOutOps(opId));
      oss << " and control dependency outs are ";
      util::append(oss, controlDependencyOutOps(opId));
      oss << ". Control dependencies should be unique, and distinct from "
             "data deps. ";
      throw error(oss.str());
    }
  }
  return outs;
}

OpIds Graph::allInOps(OpId opId) const {
  OpIds ins = dataDependencyInOps(opId);
  for (auto x : controlDependencyInOps(opId)) {
    if (std::find(ins.cbegin(), ins.cend(), x) == ins.cend()) {
      ins.push_back(x);
    } else {
      std::ostringstream oss;
      oss << "Error in allInOps for " << op(opId)
          << ", where data dependency ins are ";
      util::append(oss, dataDependencyInOps(opId));
      oss << " and control dependency ins are ";
      util::append(oss, controlDependencyInOps(opId));
      oss << ". Control dependencies should be unique, and distinct from "
             "data deps. ";
      throw error(oss.str());
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

  verifyValidAtSchedulableLevel(newId);

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

  for (auto previous :
       mayBeFinals(subGraphId(opId), NoAdditionalFwdEdges())) {
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

  for (auto inId : op(after).inTensorIds()) {
    if (inId.opId() == before) {
      // constraint already exists as a data dependency.
      return;
    }
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

std::vector<OpIds> Graph::subGraphPartitioned(const OpIds &opIds) const {
  std::vector<OpIds> p(nSubGraphs());
  for (auto id : opIds) {
    p[subGraphId(id).get_u64()].push_back(id);
  }
  return p;
}

namespace {
class InfoGetter : public poprithms::schedule::scc::NodeInfoGetter {
  const Graph &g_;
  const AdditionalFwdEdges &afe_;
  const FwdEdgeMap &fem_;

public:
  InfoGetter(const Graph &g,
             const FwdEdgeMap &fem,
             const AdditionalFwdEdges &afe)
      : g_(g), afe_(afe), fem_(fem) {}

  std::string nodeString(uint64_t n) const {
    return g_.multioutOp(fem_.opId(n)).str();
  }

  std::string edgeString(uint64_t f, uint64_t t) const final {
    OpId from = fem_.opId(f);
    OpId to_  = fem_.opId(t);
    if (g_.schedulableOp(from).isControlDependencyOutOp(to_)) {
      return "control";
    }

    for (auto o : g_.outTensorIds(f)) {
      for (auto c : g_.consumptionIds(o)) {
        if (c.opId() == t) {
          return "data";
        }
      }
    }
    if (afe_.isEdge(f, t)) {
      return "additional";
    }

    return "derived";
  }

  bool providesEdgeStrings() const final { return true; }
};

[[noreturn]] void cycleError(const Graph &g,
                             const FwdEdgeMap &fem,
                             const AdditionalFwdEdges &afe) {
  using namespace poprithms::schedule::scc;

  // TODO(T39536) when there are links, this will need to be handled
  // differently. See for example the shift scheduler.
  auto s = getSummary(
      fem.fwdEdgesCompact(), InfoGetter(g, fem, afe), IncludeSingletons::No);
  throw error(s);
}

std::vector<uint64_t> getVanillaSchedule(const Graph &g,
                                         const FwdEdgeMap &fem,
                                         const AdditionalFwdEdges &afe) {

  using namespace schedule::vanilla;
  auto sched = getSchedule_u64(
      fem.fwdEdgesCompact(), ErrorIfCycle::No, VerifyEdges::Yes);

  if (sched.size() == fem.nOps()) {
    return sched;
  }

  cycleError(g, fem, afe);
}

std::vector<uint64_t> getRandomSchedule(const Graph &g,
                                        const FwdEdgeMap &fem,
                                        const AdditionalFwdEdges &afe,
                                        uint32_t seed) {

  using namespace schedule::vanilla;
  auto sched = Scheduler<uint64_t, double>::random(fem.fwdEdgesCompact(),
                                                   {},
                                                   {},
                                                   seed,
                                                   ErrorIfCycle::Yes,
                                                   VerifyEdges::Yes);

  if (sched.size() == fem.nOps()) {
    return sched;
  }

  cycleError(g, fem, afe);
}
} // namespace

OpIds Graph::vanillaSchedule(const AdditionalFwdEdges &ae) const {
  const auto fem = getForwardEdgeMap_u64(ae);
  auto sched     = getVanillaSchedule(*this, fem, ae);
  return fem.unpacked(sched);
}

OpIds Graph::vanillaSubGraphSchedule(SubGraphId sgId,
                                     const AdditionalFwdEdges &ae) const {
  const auto fem = getSubGraphForwardEdgeMap_u64(sgId, ae);
  auto sched     = getVanillaSchedule(*this, fem, ae);
  return fem.unpacked(sched);
}

OpIds Graph::randomSchedule(uint32_t s, const AdditionalFwdEdges &ae) const {
  const auto fwdEdgeMap = getForwardEdgeMap_u64(ae);
  return fwdEdgeMap.unpacked(getRandomSchedule(*this, fwdEdgeMap, ae, s));
}

OpIds Graph::randomSubGraphSchedule(SubGraphId sgId,
                                    uint32_t s,
                                    const AdditionalFwdEdges &ae) const {
  const auto fwdEdgeMap = getSubGraphForwardEdgeMap_u64(sgId, ae);
  return fwdEdgeMap.unpacked(getRandomSchedule(*this, fwdEdgeMap, ae, s));
}

std::vector<OpIds>
Graph::vanillaSchedules(const AdditionalFwdEdges &ae) const {
  return subGraphPartitioned(vanillaSchedule(ae));
}

std::vector<OpIds>
Graph::randomSchedules(uint32_t s, const AdditionalFwdEdges &ae) const {
  return subGraphPartitioned(randomSchedule(s, ae));
}

OpIds Graph::controlDependencyInOps(OpId opId) const {
  return op(opId).controlDependencyInOps();
}

OpIds Graph::controlDependencyOutOps(OpId opId) const {
  return op(opId).controlDependencyOutOps();
}

FwdEdgeMap
Graph::getSparseForwardEdgeMap_u64(const OpIds &opIds,
                                   const AdditionalFwdEdges &ae) const {

  BiDirEdgeMap nonControlEdges;

  // Data dependencies.
  for (auto id : opIds) {
    for (auto out : dataDependencyOutOps(id)) {
      nonControlEdges.insert(id, out);
    }
  }

  // Additional dependencies.
  for (const auto &[f, t] : ae.fwdEdges()) {
    nonControlEdges.insert(f, t);
  }

  // Derived graph class dependencies.
  for (const auto &[from, tos] :
       schedulableDerivedSpecificConstraints(opIds)) {
    for (auto to : tos) {
      nonControlEdges.insert(from, to);
    }
  }

  // Control dependencies. The complexity in the following code is for
  // handling the constraint-phobic ops: we must transfer all the control
  // dependency constraints which start or end at constraint-phobic ops.
  //
  // First, we initialize the control dependency edges from this graph:

  BiDirEdgeMap nxtControlEdges;
  BiDirEdgeMap allStartEdges = nonControlEdges;
  for (auto f : opIds) {
    for (auto t : op(f).controlDependencyOutOps()) {
      nxtControlEdges.insert(f, t);
      allStartEdges.insert(f, t);
    }
  }

  // Second, while there is a control dependency involving a control
  // dependency, we transfer it:
  BiDirEdgeMap controlEdges;

  bool constraintPhobicOpHandled{true};
  while (constraintPhobicOpHandled) {

    std::swap(controlEdges, nxtControlEdges);
    nxtControlEdges = BiDirEdgeMap{};

    constraintPhobicOpHandled = false;
    for (const auto &[f, t] : controlEdges.fwdEdges()) {

      // If the constraint has a starting op which is constraints phobic, then
      // slide the start back to all inputs. What we're guaranteeing here is
      // that all indirect constraints registered so far will still be
      // satisfied after the transfers.
      if (op(f).isConstraintPhobic()) {
        constraintPhobicOpHandled = true;
        if (f != t) {
          for (auto fPrime : allStartEdges.bwdEdges(f)) {
            nxtControlEdges.insert(fPrime, t);
          }
        }
      }

      // If the constraint has an ending op which is constraints phobic, then
      // slide the end forward to all its outputs. What we're guaranteeing
      // here is that all indirect constraints registered so far will still be
      // satisfied after the transfers.
      else if (op(t).isConstraintPhobic()) {
        constraintPhobicOpHandled = true;
        for (auto tPrime : allStartEdges.fwdEdges(t)) {
          nxtControlEdges.insert(f, tPrime);
        }
      }

      // what is a control edge is constraint phobic at both start and end?
      // The end will be transferred in a subsequent iteration of this
      // while-loop.

      else {
        nxtControlEdges.insert(f, t);
      }
    }
  }

  // No control dependencies start or end at constraint-phobic ops now. How do
  // we know this while loop will ever terminate? Because the graph is a dag,
  // and all transfers from constraint-phobic starts towards the beginning of
  // the dag and and constraint-phobic ends towards the dag's end.

  FwdEdgeMap fwdEdgeMap(opIds);
  for (const auto &m : {nonControlEdges, controlEdges}) {
    for (auto [f, t] : m.fwdEdges()) {
      fwdEdgeMap.insertEdge(f, t);
    }
  }

  return fwdEdgeMap;
}

FwdEdgeMap Graph::getForwardEdgeMap_u64(const AdditionalFwdEdges &ae) const {
  return getSparseForwardEdgeMap_u64(common::multiout::Graph::opIds(), ae);
}

FwdEdgeMap
Graph::getSubGraphForwardEdgeMap_u64(SubGraphId sgId,
                                     const AdditionalFwdEdges &ae) const {
  return getSparseForwardEdgeMap_u64(opIds(sgId), ae);
}

SubGraphIds Graph::subGraphIds(const TensorIds &ids) const {
  return SubGraphId::subGraphIds(*this, ids);
}

// We could consider keeping an additional datastructre (a field in
// SubGraphState) which partitions OpIds by Graph, but in my (PopART)
// experience it's not often that this partition is needed.
OpIds Graph::opIds(SubGraphId subGraphId_) const {
  OpIds opIds_;
  for (auto id : subGraphStates.at(subGraphId_.get_u64()).ops()) {
    if (isLive(id)) {
      opIds_.push_back(id);
    }
  }
  return opIds_;
}

SubGraphId Graph::subGraphId(const TensorId &id) const {
  return subGraphId(id.opId());
}

SubGraphId Graph::subGraphId(OpId i) const { return op(i).subGraphId(); }

OpIds Graph::mayBeFinals(SubGraphId subGraphId,
                         const AdditionalFwdEdges &ae) const {
  OpIds tailEnders;
  const auto ids = opIds(subGraphId);
  for (auto id : ids) {
    if (allOutOps(id).size() == 0 && ae.isSource(id) == 0) {
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
Graph::getSchedulableColumns(const OpIds &opIds,
                             const util::StringColumn::Parameters &p) const {

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

  cols.push_back({"Graph", std::move(sg__), p});
  if (hasNonDataIns) {
    cols.push_back({"NonDataIns", std::move(nonDataIns__), p});
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

void Graph::verifyValidAtSchedulableLevel(OpId opId) const {

  // sub-graphs of inputs must match op.
  for (const auto t : inTensorIds(opId)) {
    if (subGraphId(t) != subGraphId(opId)) {
      std::ostringstream oss;
      oss << "Invalid sub-graphs for inputs to " << op(opId)
          << ". All inputs must be in the same sub-graph as the ops "
          << "sub-graph, " << subGraphId(opId);
      throw error(oss.str());
    }
  }

  // cannot have a constraint between ops in different graphs.
  for (auto out_ : allOutOps(opId)) {
    if (subGraphId(out_) != subGraphId(opId)) {
      std::ostringstream oss;
      oss << "There is a topological constraint, " << opId << " -> " << out_
          << ", but these 2 ops are not in the same sub-graph. "
          << " Correctness verification failed. ";
      throw error(oss.str());
    }
  }

  // agreement on the 2 ends of the constraint.
  for (auto opBefore : controlDependencyInOps(opId)) {
    auto afters = controlDependencyOutOps(opBefore);
    if (std::find(afters.cbegin(), afters.cend(), opId) == afters.cend()) {
      std::ostringstream oss;
      oss << "The op " << op(opId) << " has " << op(opBefore)
          << " as an input control dependency, but " << op(opBefore)
          << " does not have " << op(opId)
          << " as an output control dependency.";
      throw error(oss.str());
    }
  }
}

void Graph::verifyMultioutDerivedGraphValid() const {

  for (auto opId : multiout::Graph::opIds()) {
    verifyValidAtSchedulableLevel(opId);
  }

  verifySchedulableDerivedGraphValid();
}

void Graph::verifyMultioutDerivedOpValid(OpId opId) const {
  verifyValidAtSchedulableLevel(opId);
  verifySchedulableDerivedOpValid(opId);
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

void AdditionalFwdEdges::noWeakVTables() {
  throw error(error::error::weakVTableMessage());
}

std::vector<std::pair<OpId, OpId>> NoAdditionalFwdEdges::fwdEdges() const {
  return {};
}

} // namespace schedulable
} // namespace common
} // namespace poprithms
