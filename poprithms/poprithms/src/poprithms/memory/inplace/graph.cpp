// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "ops.hpp"

#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>

#include <memory/inplace/error.hpp>

#include <poprithms/memory/inplace/color.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/schedule/scc/scc.hpp>
#include <poprithms/schedule/transitiveclosure/partitionedtransitiveclosure.hpp>
#include <poprithms/schedule/vanilla/vanilla.hpp>
#include <poprithms/util/copybyclone_impl.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

Graph::~Graph() = default;

namespace {

void assertNonNegative(const std::vector<int64_t> &vs) {
  for (auto v : vs) {
    if (v < 0) {
      std::ostringstream oss;
      oss << "Failure in assertNonNegative, for input ";
      util::append(oss, vs);
      throw error(oss.str());
    }
  }
}
} // namespace

TensorId Graph::pad(const TensorId &id,
                    const std::array<std::vector<int64_t>, 2> &lowerAndUpper,
                    bool paddingIsParallelWriteable) {

  const auto &l = std::get<0>(lowerAndUpper);
  assertNonNegative(l);
  std::vector<uint64_t> l_u64{l.cbegin(), l.cend()};

  const auto &u = std::get<1>(lowerAndUpper);
  assertNonNegative(u);
  std::vector<uint64_t> u_u64{u.cbegin(), u.cend()};

  const auto cp =
      paddingIsParallelWriteable ? ConstantPadding::No : ConstantPadding::Yes;

  const auto sp = paddingIsParallelWriteable ? BroadcastPadding::No
                                             : BroadcastPadding::Yes;

  return pad(id, LowerPadding(l_u64), UpperPadding(u_u64), cp, sp);
}

std::vector<std::array<TensorId, 2>>
Graph::createBroadcastPadElements(const Shape &shape,
                                  const LowerPadding &l,
                                  const UpperPadding &u,
                                  ConstantPadding cp) {
  const auto alloc = cp == ConstantPadding::Yes ? constant({}) : variable({});
  const auto padShapes = shape.getPadShapes(l.get(), u.get());
  std::vector<std::array<TensorId, 2>> paddings;
  paddings.reserve(shape.rank_u64());
  for (auto [l_, u_] : padShapes) {
    paddings.push_back({expand(alloc, l_), expand(alloc, u_)});
  }
  return paddings;
}

std::vector<std::array<TensorId, 2>>
Graph::createNonAliasedPadElements(const Shape &shape,
                                   const LowerPadding &l,
                                   const UpperPadding &u,
                                   ConstantPadding cp) {
  std::vector<std::array<TensorId, 2>> paddings;
  paddings.reserve(shape.rank_u64());
  for (auto [l_, u_] : shape.getPadShapes(l.get(), u.get())) {
    if (cp == ConstantPadding::Yes) {
      paddings.push_back({constant(l_), constant(u_)});
    } else {
      paddings.push_back({variable(l_), variable(u_)});
    }
  }
  return paddings;
}

TensorId Graph::pad(const TensorId &id,
                    const LowerPadding &l,
                    const UpperPadding &u,
                    ConstantPadding cp,
                    BroadcastPadding bp) {

  // This copy is necessary, as the reference is invalidated in concat.
  const auto sh = shape(id);

  const auto paddings = (bp == BroadcastPadding::Yes)
                            ? createBroadcastPadElements(sh, l, u, cp)
                            : createNonAliasedPadElements(sh, l, u, cp);

  auto current = id;
  for (uint64_t d = 0; d < rank_u64(id); ++d) {
    current = concat(
        {std::get<0>(paddings[d]), current, std::get<1>(paddings[d])}, d);
  }
  return current;
}

std::ostream &operator<<(std::ostream &ost, CheckParallelWriteable check) {
  switch (check) {
  case CheckParallelWriteable::Yes: {
    ost << "CheckParallelWriteable::Yes";
    break;
  }

  case CheckParallelWriteable::No: {
    ost << "CheckParallelWriteable::No";
    break;
  }
  }
  return ost;
}

std::ostream &operator<<(std::ostream &ost, ConstantPadding cp) {
  switch (cp) {
  case ConstantPadding::Yes: {
    ost << "ConstantPadding::Yes";
    break;
  }

  case ConstantPadding::No: {
    ost << "ConstantPadding::No";
    break;
  }
  }
  return ost;
}

namespace {

std::vector<OpId> toOpIds(const std::vector<decltype(OpId().get())> &vals) {
  std::vector<OpId> opIds;
  opIds.reserve(vals.size());
  for (auto v : vals) {
    opIds.push_back(OpId(v));
  }
  return opIds;
}

} // namespace

std::ostream &operator<<(std::ostream &ost, const OpIds &opIds) {
  poprithms::util::append(ost, opIds);
  return ost;
}

bool Graph::satisifedWithoutAnyChange(const Constraints &constraints) const {
  for (auto [from, to] : constraints) {
    if (scheduleIndex(from) >= scheduleIndex(to)) {
      return false;
    }
  }
  return true;
}

template <class T, class... Args>
OpId Graph::createOp(const TensorIds &inIds,
                     const Shapes &outShapes,
                     Args... args) {
  return insertOp(std::make_unique<T>(
      Op::getStartingState(
          nOps_i64(), inIds, outShapes, *this), // opIds(inIds)),
      args...));
}

OpId Graph::insertOp(std::unique_ptr<Op> createdOp) {
  scheduleIsValid  = false;
  const auto inIds = createdOp->inTensorIds();
  const auto newId = insertMultioutOp(std::move(createdOp));
  for (const auto &inId : inIds) {
    op(inId.opId()).insertOut(newId);
    op(newId).insertIn(inId.opId());
  }
  op(newId).grow(aGraph(), tensorMap);
  return newId;
}

TensorId Graph::settSample(const TensorId &id, const Region &r) {
  return {createOp<SettSample>({id}, {r.nelms()}, r), 0};
}

TensorId Graph::dimShuffle(const TensorId &id, const Permutation &perm) {
  return {createOp<DimShuffle>({id}, {shape(id).dimShuffle(perm)}, perm), 0};
}

TensorId Graph::reverse(const TensorId &id, const Dimensions &d) {
  return {createOp<Reverse>({id}, {shape(id)}, d), 0};
}

TensorId Graph::reshape(const TensorId &id, const Shape &outShape) {
  return {createOp<Reshape>({id}, {outShape}), 0};
}

TensorId Graph::expand(const TensorId &id, const Shape &outShape) {
  shape(id).assertCanExpandTo(outShape);
  return {createOp<Expand>({id}, {outShape}), 0};
}

TensorId Graph::modify(const TensorId &id) {
  return {createOp<UnaryModifier>({id}, {shape(id)}), 0};
}
TensorId Graph::constant(const Shape &shape) {
  return {createOp<Alloc>({}, {shape}, ConstantColor), 0};
}

TensorId Graph::variable(const Shape &shape) {
  return {createOp<Alloc>({}, {shape}, VariableColor), 0};
}

TensorId Graph::concat(const TensorIds &ids, uint64_t axis) {
  const auto shapes_ = shapes(ids);
  Shape::assertConcattable(shapes_, axis);
  return {createOp<Concat>(ids, {Shape::concat(shapes_, axis)}, axis), 0};
}

void Graph::constraint(const OpId before, const OpId after) {
  op(before).insertOut(after);
  op(after).insertIn(before);
  // If 'after' appears before 'before' in current schedule, label invalid.
  scheduleIsValid =
      scheduleIsValid && (scheduleIndex(after) > scheduleIndex(before));
}

bool Graph::multiOutTypeSpecificEqualTo(
    const common::multiout::Graph &rhs_) const {
  // As the state of the base class completely defines the base of this class,
  // there are no additional comparisons required.
  const Graph &rhs = static_cast<const Graph &>(rhs_);
  (void)rhs;
  return true;
}

const Op &Graph::op(OpId a) const {
  // We know that all Ops in this Graph can be safely cast, so no need for
  // dynamic_cast here.
  return static_cast<const Op &>(multioutOp(a));
}
Op &Graph::op(OpId a) { return static_cast<Op &>(multioutOp(a)); }

OpeningStatuses Graph::tryOpenings(const Proposals &proposals,
                                   CheckParallelWriteable check,
                                   AllowMultiGateAlias allow) {
  OpeningStatuses statuses;
  statuses.reserve(proposals.size());
  for (const auto &p : proposals) {
    statuses.push_back(tryOpening(p, check, allow));
  }
  return statuses;
}

OpeningStatuses Graph::tryOpenings0(const TensorIds &ids,
                                    CheckParallelWriteable xp,
                                    AllowMultiGateAlias allow) {
  return tryOpenings(Proposal::open0(ids), xp, allow);
}

OpeningStatuses Graph::tryOpenings0(const OpIds &ids,
                                    CheckParallelWriteable xp,
                                    AllowMultiGateAlias allow) {
  return tryOpenings(Proposal::open0(ids), xp, allow);
}

void Graph::constraints(OpId a, const OpIds &bs) {
  for (auto b : bs) {
    constraint(a, b);
  }
}

void Graph::constraints(const OpIds &as, OpId b) {
  for (auto a : as) {
    constraint(a, b);
  }
}

ConsumptionIds Graph::modifiers(const TensorId &id) const {
  std::vector<ConsumptionId> modifiers;
  for (const auto &consumer : consumptionIds(id)) {
    if (op(consumer.opId()).modifies(consumer.inIndex())) {
      modifiers.push_back(consumer);
    }
  }
  return modifiers;
}

TensorIds Graph::allAliases(const TensorId &id) const {
  const auto aliasGraphId = tensorMap.toAliasGraphId(id);
  return tensorMap.fromAliasGraphIds(aGraph().allAliases(aliasGraphId));
}

bool Graph::areAliased(const TensorId &a, const TensorId &b) const {
  const auto id0 = tensorMap.toAliasGraphId(a);
  const auto id1 = tensorMap.toAliasGraphId(b);
  return aGraph().areAliased(id0, id1);
}

uint64_t Graph::scheduleIndex(OpId id) const {
  if (!scheduleIsValid) {
    throw error("call to scheduleIndex but !scheduleIsValid");
  }
  return invSched[id.get()];
}

// Possible optimizations for the inplacing algorithm:
//
// 1) Cache aliases between calls to tryOpening (T29079)
//
// 2) use the DAG structure to reduce alias computation, and sparsify
//    constraint calculation and insertion (T29080)

const AliasGate &Graph::asAliasGate(OpId mid) const {
  auto proposedBaseOp = &op(mid);
  auto proposedAliasGateOpPtr =
      dynamic_cast<const AliasGate *>(proposedBaseOp);
  if (!proposedAliasGateOpPtr) {
    std::ostringstream oss;
    oss << "Failure to cast Op to AliasGate. This for OpId = " << mid
        << ", where the Op trying to cast to AliasGate is "
        << *proposedBaseOp;
    throw error(oss.str());
  }
  auto &aliasGate = *proposedAliasGateOpPtr;
  return aliasGate;
}

InIndex Graph::aliasGateInIndex(OpId mid) const {
  return asAliasGate(mid).inIndex();
}

bool Graph::aliasGateIsClosed(OpId mid) const {
  return asAliasGate(mid).closed();
}

// See Scott Meyers' "Effective C++"
AliasGate &Graph::asAliasGate(OpId mid) {
  return const_cast<AliasGate &>(
      static_cast<const Graph &>(*this).asAliasGate(mid));
}

OpeningResult Graph::tryOpeningPartial(const Proposal &p,
                                       CheckParallelWriteable check,
                                       AllowMultiGateAlias allow) {

  auto &aliasGate_ = asAliasGate(p.aliasGateId());

  if (aliasGate_.nInTensors() <= p.inIndex().get()) {
    std::ostringstream oss;
    oss << "Invalid proposal input index, " << p.inIndex()
        << ", for aliasGate with only " << aliasGate_.nInTensors()
        << " input Tensors. ";
    throw error(oss.str());
  }

  const auto inTensorToAlias  = aliasGate_.inTensorId(p.inIndex());
  const auto outTensorToAlias = aliasGate_.outTensorId(0);

  if (aliasGate_.open()) {
    return OpeningResult::alreadyOpen();
  }

  // The schedule is not kept up-to-date while the Graph is being constructed,
  // so we update it here, if necessary.
  if (!scheduleIsValid) {
    const auto fwdEdges = getFwdEdges<int64_t>({});

    setSchedule(toOpIds(poprithms::schedule::vanilla::getSchedule_i64(
        fwdEdges,
        schedule::vanilla::ErrorIfCycle::No,
        schedule::vanilla::VerifyEdges::Yes)));

    if (sched.size() != nOps()) {
      std::ostringstream oss;
      oss << "A cycle detected in Graph::tryOpeningPartial, "
          << "before the proposal " << p << " was processed. "
          << "Only " << sched.size() << " of " << nOps()
          << " were scheduled. This suggests a cycle was present "
          << "in the constructed Graph. Summary from "
             "poprithms::schedule::scc:\n";
      poprithms::schedule::scc::getSummary_i64(
          fwdEdges,
          getOpNames(),
          poprithms::schedule::scc::IncludeSingletons::No);
      throw error(oss.str());
    }
    scheduleIsValid = true;
  }

  struct AliInfo {
    TensorId id;              // The Tensor whose info is stored
    ConsumptionIds modifiers; // modifiers of Tensor "id"
    TensorIds aliases;        // aliases of Tensor "id"
  };

  // Aliases of inputs which have modifiers, under the proposal.
  //
  // Example: proposal to open aliasGate.
  //          input to aliasGate is X
  //          Y is an alias of X, and Y has a modifier, so AliInfo of Y will
  //          be in the returned vector.
  //
  //            Y         .
  //           / \        .
  //      slice   unary   .
  //        |             .
  //        X             .
  //        |             .
  //       aliasGate (proposal to make reshape_)
  //
  std::vector<AliInfo> inAliasModified;
  const auto aliasesOfInTensorToAlias = allAliases(inTensorToAlias);
  for (auto ali : aliasesOfInTensorToAlias) {
    const auto m = modifiers(ali);
    if (!m.empty()) {
      inAliasModified.push_back({ali, m, allAliases(ali)});
    }
  }

  // Aliases of outputs which have modifiers under the proposal.
  //
  // Example: proposal to open aliasGate.
  //          output of slice is Y
  //          Z is an alias of Y, and Z has a modifier, so AliInfo of Z will
  //          be in the returned vector.
  //
  //     X
  //     |
  //    aliasGate
  //     |
  //     Y
  //      \.
  //     reverse
  //        |
  //        Z
  //       /.
  //   unary
  //
  std::vector<AliInfo> outAliasModified;
  const auto aliasesOfOutTensorToAlias = allAliases(outTensorToAlias);
  for (auto ali : aliasesOfOutTensorToAlias) {
    const auto m = modifiers(ali);
    if (!m.empty()) {
      outAliasModified.push_back({ali, m, allAliases(ali)});
    }
  }

  // Open the AliasGate, let the aliases "flow"
  aliasGate_.openAt(aGraph(), tensorMap, p.inIndex());

  if (check == CheckParallelWriteable::Yes) {

    // Get all Ops which modify an alias of an input or output
    std::vector<ConsumptionId> allModifiers;
    for (auto x : inAliasModified) {
      allModifiers.insert(
          allModifiers.end(), x.modifiers.cbegin(), x.modifiers.cend());
    }
    for (auto x : outAliasModified) {
      allModifiers.insert(
          allModifiers.end(), x.modifiers.cbegin(), x.modifiers.cend());
    }

    // Check that no modifiers modify a Tensor with a constant or a self-alias
    // If they do, aliasGate will remain close.
    for (auto m : allModifiers) {
      const auto &op_ = op(m.opId());
      const auto tId  = tensorMap.toAliasGraphId(op_.inTensorId(m.inIndex()));
      if (aGraph().containsColor(tId, ConstantColor) ||
          aGraph().containsAliases(tId)) {
        aliasGate_.close(aGraph(), tensorMap);
        return OpeningResult::notParallelWriteable();
      }
    }
  }

  // Ensure that this inplacing does not result in any alias gates having
  // outputs with aliases to multiple inputs
  if (allow == AllowMultiGateAlias::No) {

    // All tensors which are aliased to an input/output of the alias gate
    // being opened in this proposal. These are the tensors which have
    // different aliases under the proposal.
    auto tensorsToConsider = aliasesOfInTensorToAlias;
    tensorsToConsider.insert(tensorsToConsider.end(),
                             aliasesOfOutTensorToAlias.cbegin(),
                             aliasesOfOutTensorToAlias.cend());

    // All alias gates which the tensors to consider (above) go into. We are
    // only interested here in alias gates with more than 1 input, as we're
    // looking for alias gates with inputs which alias each other.
    std::unordered_set<const AliasGate *> aliasGatesToConsider;
    for (auto tensorToConsider : tensorsToConsider) {
      for (auto c : consumptionIds(tensorToConsider)) {
        const auto aliGate = dynamic_cast<const AliasGate *>(&op(c.opId()));
        if (aliGate && aliGate->nInTensors() > 1 && aliGate->open()) {
          aliasGatesToConsider.insert(aliGate);
        }
      }
    }

    // check for inputs at closed indices which alias the input at the open
    // index.
    for (auto aliGate : aliasGatesToConsider) {
      const auto inIds     = aliGate->inTensorIds();
      const auto openIndex = aliGate->inIndex().get();
      for (uint64_t i = 0; i < aliGate->nInTensors(); ++i) {
        if (i != openIndex && areAliased(inIds[i], inIds[openIndex])) {
          aliasGate_.close(aGraph(), tensorMap);
          return OpeningResult::gateMultiInAlias();
        }
      }
    }
  }

  Constraints newConstraints;

  // The modifiers of output aliases might have new aliases: Tensors which are
  // aliased to the inputs of aliasGate. ConsumptionIds of these new aliases
  // must execute before the modifier, so that behaviour is unchanged.
  for (const auto &x : outAliasModified) {
    for (auto newAlias : setDifference(allAliases(x.id), x.aliases)) {
      for (auto c : consumptionIds(newAlias)) {
        for (auto m : x.modifiers) {
          newConstraints.push_back({c.opId(), m.opId()});
        }
      }
    }
  }

  // The modifiers of input aliases might have new aliases: Tensors which are
  // aliased to outputs of the aliasGate.  If the modifiers were scheduled
  // after aliasGate, they must be scheduled after the new aliases too, so
  // that behaviour is unchanged.
  //
  // optimization here: don't need to go
  // through all in modifiers, only the ones which might be scheduled first
  // (T29080)
  for (const auto &x : inAliasModified) {
    auto diff = setDifference(allAliases(x.id), x.aliases);
    for (auto m : x.modifiers) {
      if (scheduleIndex(m.opId()) > scheduleIndex(aliasGate_.id())) {

        // remoteinplace (popart) test requires this constraint TODO(T29862):
        // as follow-up, remove this.
        newConstraints.push_back({aliasGate_.id(), m.opId()});

        for (auto newAlias : diff) {
          for (auto c : consumptionIds(newAlias)) {
            newConstraints.push_back({c.opId(), m.opId()});
          }
        }
      }
    }
  }

  // sort and unique-ify the constraints.
  std::sort(newConstraints.begin(), newConstraints.end());
  const auto E = std::unique(newConstraints.begin(), newConstraints.end());
  newConstraints.erase(E, newConstraints.cend());

  if (satisifedWithoutAnyChange(newConstraints)) {
    return OpeningResult::validWithUnchangedSchedule(
        std::move(newConstraints));
  }

  auto proposedSchedule =
      toOpIds(poprithms::schedule::vanilla::getSchedule_i64(
          getFwdEdges<int64_t>(newConstraints),
          schedule::vanilla::ErrorIfCycle::No,
          schedule::vanilla::VerifyEdges::No));

  if (proposedSchedule.size() != nOps()) {
    aliasGate_.close(aGraph(), tensorMap);
    return OpeningResult::cycle();
  }

  return OpeningResult::validWithChangedSchedule(std::move(newConstraints),
                                                 std::move(proposedSchedule));
}

OpeningStatus Graph::tryOpening(const Proposal &p,
                                const CheckParallelWriteable check,
                                const AllowMultiGateAlias allow) {

  auto partial = tryOpeningPartial(p, check, allow);
  if (partial.status() == OpeningStatus::Valid) {
    completeOpening(partial);
  }
  return partial.status();
}

void Graph::completeOpening(const OpeningResult &r) {
  if (r.status() != OpeningStatus::Valid) {
    std::ostringstream oss;
    oss << "Invalid call to completeStatus with OpeningResult " << r
        << ". Status must be Valid.";
    throw error(oss.str());
  }
  if (r.scheduleChange()) {
    auto sch = r.schedule();
    setSchedule(std::move(sch));
  }
  constraints(r.constraints());
  scheduleIsValid = true;
}

void Graph::backoutOpening(const Proposal &proposal) {
  asAliasGate(proposal.aliasGateId()).close(aGraph(), tensorMap);
}

void Graph::setSchedule(OpIds &&schedule_) {
  sched = std::move(schedule_);
  invSched.resize(nOps(), std::numeric_limits<uint64_t>::max());
  for (uint64_t i = 0; i < sched.size(); ++i) {
    invSched[sched[i].get()] = i;
  }
}

void Graph::constraints(const Constraints &constraints_) {
  for (const auto &[from_, to_] : constraints_) {
    constraint(from_, to_);
  }
}

template <typename T>
std::vector<std::vector<T>>
Graph::getFwdEdges(const Constraints &additional) const {
  Edges<T> fwdEdges(nOps());
  for (uint64_t i = 0; i < nOps(); ++i) {
    auto outs_ = op(i).outs();
    fwdEdges[i].reserve(outs_.size());
    for (auto o : outs_) {
      fwdEdges[i].push_back(static_cast<T>(o.get()));
    }
  }
  for (const auto &[from, to] : additional) {
    fwdEdges[static_cast<uint64_t>(from.get())].push_back(
        static_cast<T>(to.get()));
  }
  return fwdEdges;
}

template <typename T, typename Conditional>
Graph::Edges<T> Graph::getConditionalFwdEdges(Conditional &&condition) const {
  Edges<T> fwdEdges(nOps());
  for (uint64_t from = 0; from < nOps(); ++from) {
    fwdEdges[from].reserve(op(from).nOuts());
    for (auto to : op(from).outs()) {
      if (condition(from, to)) {
        fwdEdges[from].push_back(static_cast<T>(to.get()));
      }
    }
  }
  return fwdEdges;
}

void Graph::appendOpColumns(std::ostream &ost, const OpIds &opIds) const {

  const auto nTens     = nMultioutRows(opIds);
  const auto aliasedTo = aGraph().allAliases();

  auto cols = getMultioutColumns(opIds);

  using Strings = std::vector<std::string>;

  // extensions:
  Strings tensorId__(nTens, "");
  Strings inOps__(nTens, "");
  Strings tensorType__(nTens, "");
  Strings selfAliases__(nTens, "");
  Strings constants__(nTens, "");
  Strings aliasedTo__(nTens, "");

  uint64_t ti = 0;
  for (auto opId : opIds) {
    inOps__[ti] = util::getStr(op(opId).ins());
    for (uint64_t o = 0; o < op(opId).nOutTensors(); ++o) {
      const auto aliasId = tensorMap.toAliasGraphId(op(opId).outTensorId(o));
      tensorId__[ti]     = std::to_string(aliasId.get());
      tensorType__[ti]   = aGraph().typeString(aliasId);
      selfAliases__[ti]  = aGraph().containsAliases(aliasId) ? "yes" : "no";
      constants__[ti] =
          aGraph().containsColor(aliasId, ConstantColor) ? "yes" : "no";
      aliasedTo__[ti] = getStr(aliasedTo[aliasId.get()]);
      ++ti;
    }
    if (op(opId).nOutTensors() == 0) {
      ++ti;
    }
  }

  cols.push_back({"TensorId", tensorId__});
  cols.push_back({"InOps", inOps__});
  cols.push_back({"Type", tensorType__});

  // Only add logging about which Tensors self-alias if any of them actually
  // do.
  if (std::any_of(
          selfAliases__.cbegin(), selfAliases__.cend(), [](const auto &x) {
            return x.find("yes") != std::string::npos;
          })) {
    cols.push_back({"SelfAliases", aliasedTo__});
  }

  // Only add logging about which Tensors contain constants if any of them
  // actually do.
  if (std::any_of(
          constants__.cbegin(), constants__.cend(), [](const auto &x) {
            return x.find("yes") != std::string::npos;
          })) {
    cols.push_back({"Constant", constants__});
  }

  ost << alignedColumns(cols);
}

std::ostream &operator<<(std::ostream &ost, const Graph &g) {
  g.append(ost);
  return ost;
}

OpId Graph::multi(const TensorIds &inIds,
                  const Shapes &outShapes,
                  const CrossLinks &mapping) {
  return createOp<Multi>(inIds, outShapes, mapping);
}

TensorId Graph::aliasGate(const TensorIds &ids) {
  return {createOp<AliasGate>(ids, {Shape::numpyVariadic(shapes(ids))}), 0};
}

bool Graph::isAliasGate(OpId id) const {
  const auto &op_ = op(id);
  return (dynamic_cast<const AliasGate *>(&op_) != nullptr);
}

TensorId Graph::aliasGate(const TensorIds &ids, InIndex inInd) {
  if (inInd < 0) {
    return aliasGate(ids);
  }

  const auto outShape = Shape::numpyVariadic(shapes(ids));
  return {createOp<AliasGate>(ids, {outShape}, inInd), 0};
}

TensorId Graph::slice(const TensorId &id, const Lower &l, const Upper &u) {
  return settSample(id, Region::fromBounds(shape(id), l, u));
}

TensorId Graph::subSample(const TensorId &id, Stride s, Dimension d) {
  return settSample(id, Region::fromStride(shape(id), s, d));
}

TensorId Graph::subSample(const TensorId &id, const Strides &strides) {
  return settSample(id, Region::fromStrides(shape(id), strides));
}

TensorId Graph::flatten(const TensorId &id) {
  return reshape(id, {shape(id).nelms()});
}

bool Graph::mightContainAmbiguity() const {

  // For performance reasons, we first search for an ambiguity with a
  // carefully chosen subset of all edges (constraints). If no ambiguity is
  // found, we can be sure that there is no ambiguity with the full set of
  // edges, as more edges implies less scheduling freedom. If there is an
  // ambiguity found, we must check for an ambiguity with the full set of
  // edges. This 'trick' to accelerate the ambiguity search assumes that in
  // most cases, there is no ambiguity.
  //
  //
  // 1) What is the "specially chosen" subset of edges, and why might this
  // subset be a good proxy for the full set of edges?
  // -----------
  // We do not include an edge a->b if it is a data dependency where none of
  // the outputs of a are aliased to any outputs of b. An example of where
  // this might result in a false positive (detecting an ambiguity where there
  // is none) is:
  //
  // a -> m0 (data constraint, m0 is a modifier)
  // a -> m1 (data constraint, m1 is a modifier)
  // m0 -> b (data constraint, with no aliases between outputs of m0 and
  //          outputs of b)
  // b -> m1 (a non-data constraint).
  //
  // In this case, the proxy edge set does not contain m0->b. The presence of
  // edge m0 -> b ensures there is no ambiguity when considering the full set
  // of edges, but its exclusion from the proxy set means
  // `mightContainAmbiguity` returns `true` even though there is no ambiguity.
  //
  // Diagramatically:
  //
  // ----> data constraint
  // ....> non-data contraint.
  //
  //
  //       the edge which is missing
  //       with the 'proxy' edge set
  //                  v
  //       +---> m0 ----> aliasGate(b)
  //       |                  .
  //  a >--+                  .
  //       |                  .
  //       +--->  m1  <.......+
  //
  //
  // 2) Why do we think this subset is a good choice for performance reasons?
  // -----------
  // The PartitionedTransitiveClosure is quadratic in memory in the size of
  // its largest connected component. By removing edges, it becomes more
  // likely for large components to be divided into smaller components.
  //
  //
  // It's worth repeating that this is a proxy purely for performance reasons:
  // if an ambiguity is detected, then the full edge graph will be used.
  //
  // The inclusion condition:
  auto inclusionCondition = [this](OpId from, OpId to) {
    const auto toIns = op(to).inTensorIds();

    // Non-data dependency.
    if (std::all_of(toIns.cbegin(), toIns.cend(), [from](auto tId) {
          return tId.opId() != from;
        })) {
      return true;
    }

    // Non-aliasing data dependency.
    for (auto t0 : op(from).outTensorIds()) {
      for (auto t1 : op(to).outTensorIds()) {
        if (areAliased(t0, t1)) {
          return true;
        }
      }
    }
    // aliasing data dependency.
    return false;
  };

  return containsAmbiguity(
      getConditionalFwdEdges<uint64_t>(inclusionCondition));
}

bool Graph::containsAmbiguity() const {

  if (!mightContainAmbiguity()) {
    return false;
  }

  // An ambiguity was found with the reduced edge map, see if there is still
  // one with the full edge map.
  return definitelyContainsAmbiguity();
}

bool Graph::containsAmbiguity(
    const std::vector<std::vector<uint64_t>> &edges) const {

  // We look for 2 modifiers, which modify the same data, but are
  // unconstrained, relative to each other.
  //
  // The TransitiveClosure class (and the PartitionedTransitiveClosure class)
  // is used for querying whether any 2 Ops in a graph are relatively
  // unconstrained.

  schedule::transitiveclosure::PartitionedTransitiveClosure ptc(edges);

  /// For every modifying consumer #m0, of every Tensor #t0,
  for (auto t0 : tensorIds()) {
    auto mods0 = modifiers(t0);
    if (!mods0.empty()) {

      /// and for all aliases of #t0, called #t1, and modifiers #m1 of #t1,
      for (auto t1 : allAliases(t0)) {
        auto mods1 = modifiers(t1);
        for (auto m0 : mods0) {
          for (auto m1 : mods1) {

            /// Are #m0 and #m1 relatively unconstrained? If so, the graph
            /// contains an ambiguity.
            if (m0.opId() != m1.opId()) {
              if (ptc.unconstrainedInBothDirections(m0.opId().get(),
                                                    m1.opId().get())) {
                return true;
              }
            }
          }
        }
      }
    }
  }
  return false;
}

void Graph::multiOutTypeSpecificRemoveOp(
    OpId opToRemove,
    const OptionalTensorIds &outputSubstitutes) {
  // TODO(T49501)
  std::ostringstream oss;
  oss << "Must implement multiOutTypeSpecificRemoveOp. Called with "
         "opToRemove = "
      << opToRemove << " and outputSubstitutes = " << outputSubstitutes
      << ".";
  throw error(oss.str());
}

void Graph::multiOutTypeSpecificVerifyValidOutputSubstitute(
    const TensorId &before,
    const TensorId &after) const {
  // TODO(T49501)
  std::ostringstream oss;
  oss << "Must implement multiOutTypeSpecificVerifyValidOutputSubstitute. "
      << "Called with before = " << before << " and after = " << after
      << ". ";
  throw error(oss.str());
}

} // namespace inplace
} // namespace memory
} // namespace poprithms
