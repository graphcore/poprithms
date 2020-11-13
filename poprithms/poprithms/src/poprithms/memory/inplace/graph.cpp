// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "ops.hpp"
#include "poprithms/schedule/scc/scc.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <numeric>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

#include <poprithms/memory/alias/graph.hpp>
#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/schedule/vanilla/vanilla.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

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

TensorId Graph::concat_(const TensorIds &ins, uint64_t axis) {
  return concat(ins, AliasType::allInplace(), axis);
}

TensorId Graph::settSample_(const TensorId &id, const Region &w) {
  return settSample(id, AliasType::allInplace(), w);
}

TensorId Graph::slice(const TensorId &tIn,
                      AliasType t,
                      const Lower &l,
                      const Upper &u) {
  return settSample(tIn, t, Region::fromBounds(shape(tIn), l, u));
}
TensorId Graph::slice_(const TensorId &tIn, const Lower &l, const Upper &u) {
  return slice(tIn, AliasType::allInplace(), l, u);
}

TensorId
Graph::subSample_(const TensorId &tIn, int64_t stride, uint64_t dimension) {
  return subSample(tIn, AliasType::allInplace(), stride, dimension);
}

TensorId Graph::reverse_(const TensorId &tIn, const Dimensions &dims) {
  return reverse(tIn, AliasType::allInplace(), dims);
}

TensorId Graph::reshape_(const TensorId &id, const Shape &shape) {
  return reshape(id, AliasType::allInplace(), shape);
}

TensorId Graph::identity_(const TensorId &tIn) {
  return identity(tIn, AliasType::allInplace());
}

TensorId Graph::dimShuffle_(const TensorId &id, const Permutation &perm) {
  return dimShuffle(id, AliasType::allInplace(), perm);
}

TensorId Graph::expand_(const TensorId &id, const Shape &shape) {
  return expand(id, AliasType::allInplace(), shape);
}

TensorId Graph::unary_(const TensorId &inTensor) {
  return unary(inTensor, AliasType::allInplace());
}

OpIds Graph::getOpIds(const TensorIds &tensorIds) {
  OpIds opIds_;
  opIds_.reserve(tensorIds.size());
  for (const auto &tid : tensorIds) {
    opIds_.push_back(tid.opId());
  }
  return opIds_;
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

std::vector<decltype(OpId().get())>
fromOpIds(const std::vector<OpId> &opIds) {
  std::vector<decltype(OpId().get())> vals;
  vals.reserve(opIds.size());
  for (auto opId : opIds) {
    vals.push_back(opId.get());
  }
  return vals;
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

uint64_t Graph::nInTensors(OpId id) const { return op(id).nInTensors(); }

namespace {

Op::State getBaseState(const OpId opId,
                       const TensorIds &inIds,
                       const Shapes &outShapes,
                       const AliasType aType,
                       const OpIds &opIns) {

  const OpIds opOuts{};
  const std::string name{};
  std::vector<Consumers> consumers(outShapes.size());

  return Op::State(
      opId, opIns, opOuts, inIds, consumers, outShapes, aType, name);
}
} // namespace

template <class T, class... Args>
OpId Graph::createOp(const TensorIds &inIds,
                     const Shapes &outShapes,
                     const AliasType aType,
                     Args... args) {

  scheduleIsValid = false;

  auto createdOp = std::make_unique<T>(
      getBaseState(nOps_i64(), inIds, outShapes, aType, getOpIds(inIds)),
      args...);

  for (uint64_t inIndex = 0; inIndex < inIds.size(); ++inIndex) {
    const auto inTensor = inIds[inIndex];
    const auto sourceId = inTensor.opId();
    auto &source        = op(sourceId);
    source.insertConsumer(inTensor.outIndex(),
                          {createdOp->id(), InIndex(inIndex)});
  }
  ops.emplace_back(std::move(createdOp));
  ops.back().up->grow(aGraph(), tensorMap);
  auto newId = ops.back().up->id();

  return newId;
}

TensorId Graph::concat(const TensorIds &inIds,
                       const AliasType aType,
                       const uint64_t axis) {
  const auto outShape = Shape::concat(shapes(inIds), axis);
  const auto opId     = createOp<Concat>(inIds, {outShape}, aType, axis);
  return {opId, OutIndex(0)};
}

TensorId Graph::unary(const TensorId &inId, const AliasType aType) {
  const auto opId = createOp<Unary>({inId}, {shape(inId)}, aType);
  return {opId, OutIndex(0)};
}

std::vector<std::array<TensorId, 2>>
Graph::createBroadcastPadElements(const Shape &s,
                                  const LowerPadding &l,
                                  const UpperPadding &u,
                                  ConstantPadding cp) {
  const auto alloc = cp == ConstantPadding::Yes ? constant({}) : variable({});
  const auto padShapes = s.getPadShapes(l.get(), u.get());
  std::vector<std::array<TensorId, 2>> paddings;
  paddings.reserve(s.rank_u64());
  for (auto [l_, u_] : padShapes) {
    paddings.push_back({expand_(alloc, l_), expand_(alloc, u_)});
  }
  return paddings;
}

std::vector<std::array<TensorId, 2>>
Graph::createNonAliasedPadElements(const Shape &s,
                                   const LowerPadding &l,
                                   const UpperPadding &u,
                                   ConstantPadding cp) {
  std::vector<std::array<TensorId, 2>> paddings;
  paddings.reserve(s.rank_u64());

  for (auto [l_, u_] : s.getPadShapes(l.get(), u.get())) {
    if (cp == ConstantPadding::Yes) {
      paddings.push_back({constant(l_), constant(u_)});
    } else {
      paddings.push_back({variable(l_), variable(u_)});
    }
  }

  return paddings;
}

TensorId Graph::pad_(const TensorId &inId,
                     const LowerPadding &l,
                     const UpperPadding &u,
                     ConstantPadding cp,
                     BroadcastPadding bp) {
  auto paddings = (bp == BroadcastPadding::Yes)
                      ? createBroadcastPadElements(shape(inId), l, u, cp)
                      : createNonAliasedPadElements(shape(inId), l, u, cp);
  TensorId current = inId;
  for (uint64_t d = 0; d < rank_u64(inId); ++d) {
    current = concat_(
        {std::get<0>(paddings[d]), current, std::get<1>(paddings[d])}, d);
  }
  return current;
}

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

TensorId Graph::pad_(const TensorId &inTensor,
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

  return pad_(inTensor, LowerPadding(l_u64), UpperPadding(u_u64), cp, sp);
}

TensorId
Graph::binary(const TensorId &arg0, const TensorId &arg1, AliasType t) {
  const auto outShape = shape(arg0).numpyBinary(shape(arg1));
  const auto opId     = createOp<Binary>({arg0, arg1}, {outShape}, t);
  return {opId, OutIndex(0)};
}

TensorId
Graph::reshape(const TensorId &inId, AliasType t, const Shape &outShape) {
  if (outShape.nelms_u64() != shape(inId).nelms_u64()) {
    std::ostringstream oss;
    oss << "Invalid reshape, number of elements changes. "
        << "Cannot reshape from " << shape(inId) << " to " << outShape
        << ". ";
    throw error(oss.str());
  }
  const auto opId = createOp<Reshape>({inId}, {outShape}, t);
  return {opId, OutIndex(0)};
}

TensorId Graph::flatten(const TensorId &id, AliasType t) {
  return reshape(id, t, shape(id).flatten());
}

OpId Graph::multi(const TensorIds &inIds,
                  const Shapes &outShapes,
                  const Multi::Mapping &mapping) {
  const auto opId =
      createOp<Multi>(inIds, outShapes, AliasType::none(), mapping);

  for (const auto &crossAlias : mapping) {
    const auto inShape  = shape(op(opId).inTensorId(crossAlias.in()));
    const auto outShape = shape(op(opId).outTensorId(crossAlias.out()));
    if (inShape != outShape) {
      std::ostringstream oss;
      oss << "Incompatible Shapes in Graph::multi, for CrossAlias "
          << crossAlias << ". The input shape at index " << crossAlias.in()
          << " is " << inShape << ", and the output shape at index "
          << crossAlias.out() << " is " << outShape << '.';
      throw error(oss.str());
    }
  }
  return opId;
}

TensorId
Graph::settSample(const TensorId &inId, AliasType t, const Region &r) {
  const auto opId = createOp<SettSample>({inId}, {r.nelms()}, t, r);
  return {opId, OutIndex(0)};
}

TensorId Graph::dimShuffle(const TensorId &inId,
                           AliasType t,
                           const Permutation &perm) {
  const auto opId =
      createOp<DimShuffle>({inId}, {shape(inId).dimShuffle(perm)}, t, perm);
  return {opId, OutIndex(0)};
}

TensorId Graph::subSample(const TensorId &inId,
                          AliasType t,
                          int64_t stride,
                          uint64_t dimension) {
  return settSample(
      inId,
      t,
      Region::fromStripe(shape(inId), dimension, {1, stride - 1, 0}));
}

TensorId
Graph::subSample(const TensorId &inId, AliasType t, const Strides &strides) {
  std::vector<nest::Sett> setts;
  setts.reserve(strides.size());
  for (auto stride : strides.get()) {
    setts.push_back({{{1, stride - 1, 0}}});
  }
  return settSample(inId, t, Region(shape(inId), setts));
}

TensorId Graph::subSample_(const TensorId &inId, const Strides &strides) {
  return subSample(inId, AliasType::allInplace(), strides);
}

TensorId Graph::identity(const TensorId &tIn, AliasType t) {
  const auto opId = createOp<Identity>({tIn}, {shape(tIn)}, t);
  return {opId, OutIndex(0)};
}

TensorId Graph::reverse(const TensorId &tIn,
                        AliasType t,
                        const Dimensions &dimensions) {
  const auto opId = createOp<Reverse>({tIn}, {shape(tIn)}, t, dimensions);
  return {opId, OutIndex(0)};
}

TensorId Graph::expand(const TensorId &tIn, AliasType t, const Shape &shape) {
  const auto opId = createOp<Expand>({tIn}, {shape}, t);
  return {opId, OutIndex(0)};
}

TensorId Graph::constant(const Shape &shape) {
  const auto opId =
      createOp<Alloc>({}, {shape}, AliasType::outplace(), Constant);
  return {opId, OutIndex(0)};
}

TensorId Graph::variable(const Shape &shape) {
  const auto opId =
      createOp<Alloc>({}, {shape}, AliasType::outplace(), Variable);
  return {opId, OutIndex(0)};
}

void Graph::setName(const OpId id, const std::string &name) {
  op(id).setName(name);
}

void Graph::setName(const TensorId &id, const std::string &name) {
  if (op(id.opId()).nOutTensors() != 1) {
    std::ostringstream oss;
    oss << "Cannot call Graph::setName(TensorId=" << id << ", name=" << name
        << '.' << "), because the Op creator has multiple outputs. "
        << "Call setName(OpId=" << id.opId() << ", name=" << name
        << ") instead.";
    throw error(oss.str());
  }
  setName(id.opId(), name);
}

void Graph::constraint(const OpId before, const OpId after) {
  op(before).insertOut(after);
  op(after).insertIn(before);
  if (scheduleIsValid && scheduleIndex(before) >= scheduleIndex(after)) {
    scheduleIsValid = false;
  }
}

const Op &Graph::op(OpId a) const {
  if (a.get() >= nOps_i64()) {
    std::ostringstream oss;
    oss << "The number of Ops in this Graph is " << nOps()
        << ", so there is no Op with OpId " << a << '.';
    throw error(oss.str());
  }

  const auto &opPtr = ops[static_cast<uint64_t>(a.get())].up;

  // In case we decide that we need to delete Ops for this transformation at
  // some point:
  if (!opPtr) {
    throw error("nullptr in op(" + std::to_string(a.get()) + ").");
  }
  return *opPtr;
}

// See Scott Meyers' "Effective C++"
Op &Graph::op(OpId id) {
  return const_cast<Op &>(static_cast<const Graph &>(*this).op(id));
}

Graph::UpOp::UpOp(std::unique_ptr<Op> x) : up(std::move(x)) {}

bool Graph::UpOp::operator==(const Graph::UpOp &rhs) const {
  if ((up && !rhs.up) || (!up && rhs.up)) {
    return false;
  }
  if (!up && !rhs.up) {
    return true;
  }
  return (*up == *rhs.up);
}

Graph::UpOp::UpOp() = default;

Graph::UpOp::UpOp(const Graph::UpOp &x)
    : Graph::UpOp(x.up ? x.up->clone() : nullptr) {}

Graph::UpOp::~UpOp() = default;

Graph::UpOp &Graph::UpOp::UpOp::operator=(const Graph::UpOp &x) {
  if (x.up) {
    up = x.up->clone();
  }
  return *this;
}

AliasType Graph::aliasType(OpId opId) const { return op(opId).aliasType(); }

std::vector<InplaceStatus> Graph::tryInplaces(const Proposals &proposals,
                                              CheckParallelWriteable check) {
  std::vector<InplaceStatus> statuses;
  statuses.reserve(proposals.size());
  for (const auto &p : proposals) {
    statuses.push_back(tryInplace(p, check));
  }
  return statuses;
}

std::vector<Shape> Graph::shapes(const TensorIds &ids) const {
  std::vector<Shape> shapes;
  shapes.reserve(ids.size());
  for (const auto &id : ids) {
    shapes.push_back(shape(id));
  }
  return shapes;
}

Shape Graph::shape(const TensorId &tid) const {
  return op(tid.opId()).outShape(tid.outIndex());
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

Consumers Graph::modifiers(const TensorId &id) const {
  std::vector<Consumer> modifiers;
  for (const auto &consumer : consumers(id)) {
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

const Consumers &Graph::consumers(const TensorId &id) const {
  return op(id.opId()).consumers(id.outIndex());
}

TensorIds Graph::difference(TensorIds a, TensorIds b) const {
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());

  TensorIds diff;
  diff.reserve(a.size() - b.size());

  std::set_difference(a.cbegin(),
                      a.cend(),
                      b.cbegin(),
                      b.cend(),
                      std::inserter(diff, diff.begin()));

  return diff;
}

uint64_t Graph::scheduleIndex(OpId id) const {
  if (!scheduleIsValid) {
    throw error("call to scheduleIndex but !scheduleIsValid");
  }
  return invSched[id.get()];
}

// Possible optimizations for the inplacing algorithm:
//
// 1) Cache aliases between calls to tryInplace (T29079)
//
// 2) use the DAG structure to reduce alias computation, and sparsify
//    constraint calculation and insertion (T29080)
//

InplaceResult Graph::tryInplacePartial(const Proposal &p,
                                       CheckParallelWriteable check) {

  auto &proposedOp = op(p.tensorId().opId());

  if (proposedOp.aliasType() != AliasType::outplace()) {
    return InplaceResult::alreadyInplace();
  }

  // The schedule is not kept up-to-date while the Graph is being constructed,
  // so we update it here, if necessary.
  if (!scheduleIsValid) {
    const auto fwdEdges = getFwdEdges({});
    setSchedule(toOpIds(poprithms::schedule::vanilla::getSchedule_i64(
        fwdEdges,
        schedule::vanilla::ErrorIfCycle::No,
        schedule::vanilla::VerifyEdges::Yes)));
    if (sched.size() != nOps()) {
      std::ostringstream oss;
      oss << "A cycle detected in Graph::tryInplacePartial, "
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
    TensorId id;         // The Tensor whose info is stored
    Consumers modifiers; // modifiers of Tensor "id"
    TensorIds aliases;   // aliases of Tensor "id"
  };

  // Aliases of inputs which have modifiers, under the proposal.
  //
  // Example: proposal is to make reshape inplace
  //          input to reshape is X
  //          Y is an alias of X, and Y has a modifier, so AliInfo of Y will
  //          be in the returned vector.
  //
  //            Y         .
  //           / \        .
  //      slice_  unary_  .
  //        |             .
  //        X             .
  //        |             .
  //      reshape (proposal to make reshape_)
  //
  std::vector<AliInfo> inAliasModified;
  for (auto t : proposedOp.inAliasIdsIf(p.type())) {
    for (auto ali : allAliases(t)) {
      const auto m = modifiers(ali);
      if (!m.empty()) {
        inAliasModified.push_back({ali, m, allAliases(ali)});
      }
    }
  }

  // Aliases of outputs which have modifiers under the proposal.
  //
  // Example: proposal to make slice inplace.
  //          output of slice is Y
  //          Z is an alias of Y, and Z has a modifier, so AliInfo of Z will
  //          be in the returned vector.
  //
  //     X
  //     |
  //   slice (proposal to make slice_)
  //     |
  //     Y
  //     |
  //   reverse_
  //     |
  //     Z
  //     |
  //   unary_
  //
  std::vector<AliInfo> outAliasModified;
  for (auto t : proposedOp.outAliasIdsIf(p.type())) {
    for (auto ali : allAliases(t)) {
      const auto m = modifiers(ali);
      if (!m.empty()) {
        outAliasModified.push_back({ali, m, allAliases(ali)});
      }
    }
  }

  // All Tensors which are aliased to an input Tensor of the proposed Op, if
  // the input is modified by the proposed Op.
  //
  // Example: proposal is to make unary inplace.
  //          X is an input to unary, at the input index unary_ modifies.
  //          Y is aliased to X, and so Y is in the returned vector
  //
  //            X                                          .
  //          /   \                                        .
  //  reshape_      unary (proposal to make unary_)        .
  //     |                                                 .
  //     Y                                                 .
  //
  TensorIds aliasedPreChangeToModifiedIns;
  for (auto t : proposedOp.inModifiedIdsIf(p.type())) {
    for (auto x : allAliases(t)) {
      aliasedPreChangeToModifiedIns.push_back(x);
    }
  }

  // Make proposedOp inplace.
  proposedOp.apply(aGraph(), tensorMap, p.type());

  if (check == CheckParallelWriteable::Yes) {

    // Get all Ops which modify an alias of an input or output
    std::vector<Consumer> allModifiers;
    for (auto x : inAliasModified) {
      allModifiers.insert(
          allModifiers.end(), x.modifiers.cbegin(), x.modifiers.cend());
    }
    for (auto x : outAliasModified) {
      allModifiers.insert(
          allModifiers.end(), x.modifiers.cbegin(), x.modifiers.cend());
    }
    for (auto i : proposedOp.modifyingIndices()) {
      allModifiers.push_back({proposedOp.id(), i});
    }

    // Check that no modifiers modify a Tensor with a constant or a self-alias
    // If they do, return proposedOp to be outplace.
    for (auto m : allModifiers) {
      const auto &op_ = op(m.opId());
      const auto tId  = tensorMap.toAliasGraphId(op_.inTensorId(m.inIndex()));
      if (aGraph().containsColor(tId, Constant) ||
          aGraph().containsAliases(tId)) {
        proposedOp.apply(aGraph(), tensorMap, AliasType::outplace());
        return InplaceResult::notParallelWriteable();
      }
    }
  }

  Constraints newConstraints;

  // The modifiers of output aliases might have new aliases: Tensors which are
  // aliased to the inputs of proposedOp. Consumers of these new aliases must
  // execute before the modifier, so that behaviour is unchanged.
  for (const auto &x : outAliasModified) {
    for (auto newAlias : difference(allAliases(x.id), x.aliases)) {
      for (auto c : consumers(newAlias)) {
        for (auto m : x.modifiers) {
          newConstraints.push_back({c.opId(), m.opId()});
        }
      }
    }
  }

  // The modifiers of input aliases might have new aliases: Tensors which are
  // aliased to outputs of the proposedOp.  If the modifiers were scheduled
  // after proposedOp, they must be scheduled after the new aliases too, so
  // that behaviour is unchanged.
  //
  // optimization here: don't need to go
  // through all in modifiers, only the ones which might be scheduled first
  // (T29080)
  for (const auto &x : inAliasModified) {
    auto diff = difference(allAliases(x.id), x.aliases);
    for (auto m : x.modifiers) {
      if (scheduleIndex(m.opId()) > scheduleIndex(proposedOp.id())) {

        // remoteinplace (popart) test requires this constraint TODO(T29862):
        // as follow-up, remove this.
        newConstraints.push_back({proposedOp.id(), m.opId()});

        for (auto newAlias : diff) {
          for (auto c : consumers(newAlias)) {
            newConstraints.push_back({c.opId(), m.opId()});
          }
        }
      }
    }
  }

  // Final set of constraints: Aliases of modified inputs to proposedOp must
  // have there consumers scheduled before proposedOp, to keep behaviour
  // unchanged.
  for (auto x : aliasedPreChangeToModifiedIns) {
    for (auto c : consumers(x)) {
      if (c.opId() != proposedOp.id()) {
        newConstraints.push_back({c.opId(), proposedOp.id()});
      }
    }
  }

  // sort and unique-ify the constraints.
  std::sort(newConstraints.begin(), newConstraints.end());
  const auto E = std::unique(newConstraints.begin(), newConstraints.end());
  newConstraints.erase(E, newConstraints.cend());

  if (satisifedWithoutAnyChange(newConstraints)) {
    constraints(newConstraints);
    scheduleIsValid = true;
    return InplaceResult::validWithUnchangedSchedule(
        std::move(newConstraints));
  }

  auto proposedSchedule =
      toOpIds(poprithms::schedule::vanilla::getSchedule_i64(
          getFwdEdges(newConstraints),
          schedule::vanilla::ErrorIfCycle::No,
          schedule::vanilla::VerifyEdges::No));

  if (proposedSchedule.size() != nOps()) {
    proposedOp.apply(aGraph(), tensorMap, AliasType::outplace());
    return InplaceResult::cycle();
  }

  return InplaceResult::validWithChangedSchedule(std::move(newConstraints),
                                                 std::move(proposedSchedule));
}

InplaceStatus Graph::tryInplace(const Proposal &p,
                                const CheckParallelWriteable check) {

  auto partial = tryInplacePartial(p, check);
  if (partial.status() == InplaceStatus::Valid) {
    completeInplace(partial);
  }
  return partial.status();
}

void Graph::completeInplace(const InplaceResult &inplaceResult) {
  if (inplaceResult.status() != InplaceStatus::Valid) {
    std::ostringstream oss;
    oss << "Invalid call to completeStatus with InplaceResult "
        << inplaceResult << ". Status must be Valid.";
    throw error(oss.str());
  }
  if (inplaceResult.scheduleChange()) {
    auto sch = inplaceResult.schedule();
    setSchedule(std::move(sch));
  }
  constraints(inplaceResult.constraints());
  scheduleIsValid = true;
}

void Graph::backoutInplace(const Proposal &proposal) {
  auto &proposedOp = op(proposal.tensorId().opId());
  proposedOp.apply(aGraph(), tensorMap, AliasType::outplace());
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

uint64_t Graph::nTensors() const {
  uint64_t n = 0;
  for (uint64_t i = 0; i < nOps(); ++i) {
    n += op(i).nOutTensors();
  }
  return n;
}

Graph::FwdEdges Graph::getFwdEdges(const Constraints &additional) const {
  FwdEdges fwdEdges;
  fwdEdges.reserve(nOps());
  for (uint64_t i = 0; i < nOps(); ++i) {
    fwdEdges.push_back(fromOpIds(op(i).outs()));
  }
  for (const auto &[from, to] : additional) {
    fwdEdges[from.get()].push_back(to.get());
  }
  return fwdEdges;
}

std::vector<std::string> Graph::getOpNames() const {
  std::vector<std::string> names;
  names.reserve(nOps());
  for (uint64_t i = 0; i < nOps(); ++i) {
    names.push_back(op(i).name());
  }
  return names;
}

namespace {
template <typename T> std::string getStr(const std::vector<T> &X) {
  std::ostringstream ost;
  poprithms::util::append(ost, X);
  return ost.str();
}
} // namespace

void Graph::append(std::ostream &ost) const {

  const auto aliasedTo = aGraph().allAliases();

  // Return just enough white space to get a perfect alignment of columns.
  const auto getSpace = [](uint64_t target, const std::string &ts) {
    uint64_t taken = ts.size();
    if (taken > target) {
      return std::string(" ");
    }
    return std::string(target - taken + 1, ' ');
  };

  const auto nLines = nOps() + nTensors() + 2;

  using Strings = std::vector<std::string>;
  Strings opId__(nLines, "");
  opId__[0] = "OpId";

  Strings opDebugString__(nLines, "");
  opDebugString__[0] = "Name";

  Strings opType__(nLines, "");
  opType__[0] = "OpType";

  Strings inTensors__(nLines, "");
  inTensors__[0] = "InTensors";

  Strings inOps__(nLines, "");
  inOps__[0] = "InOps";

  Strings outIndex__(nLines, "");
  outIndex__[0] = "OutIndex";

  Strings tensorId__(nLines, "");
  tensorId__[0] = "TensorId";

  Strings tensorShape__(nLines, "");
  tensorShape__[0] = "Shape";

  Strings tensorType__(nLines, "");
  tensorType__[0] = "TensorType";

  Strings selfAliases__(nLines, "");
  selfAliases__[0] = "Aliases";

  Strings constants__(nLines, "");
  constants__[0] = "Constants";

  Strings aliasedTo__(nLines, "");
  aliasedTo__[0] = "AliasedTo";

  uint64_t l = 2;
  for (uint64_t i = 0; i < nOps(); ++i) {
    opId__[l]      = std::to_string(i);
    opType__[l]    = op(i).typeString();
    inTensors__[l] = getStr(tensorMap.toAliasGraphIds(op(i).inTensorIds()));
    inOps__[l]     = getStr(op(i).ins());
    opDebugString__[l] = op(i).name();
    // ++l;
    for (uint64_t o = 0; o < op(i).nOutTensors(); ++o) {
      const auto aliasId = tensorMap.toAliasGraphId(op(i).outTensorId(o));
      // const auto aliaseTensor = aGraph().tensor(aliasId);
      outIndex__[l]    = std::to_string(o);
      tensorId__[l]    = std::to_string(aliasId.get());
      tensorShape__[l] = getStr(aGraph().shape(aliasId).get());
      tensorType__[l]  = aGraph().typeString(aliasId);
      selfAliases__[l] = aGraph().containsAliases(aliasId) ? "yes" : "no";
      constants__[l] =
          aGraph().containsColor(aliasId, Constant) ? "yes" : "no";
      aliasedTo__[l] = getStr(aliasedTo[aliasId.get()]);
      ++l;
    }
  }

  std::vector<Strings> frags{opId__,
                             opDebugString__,
                             opType__,
                             inTensors__,
                             inOps__,
                             outIndex__,
                             tensorId__,
                             tensorShape__,
                             tensorType__,
                             selfAliases__,
                             constants__,
                             aliasedTo__};

  auto getLen = [](const Strings &v) {
    return 1 + std::accumulate(v.cbegin(),
                               v.cend(),
                               0ULL,
                               [](size_t n, const std::string &x) {
                                 return std::max(n, x.size());
                               });
  };

  std::vector<uint64_t> lens;
  for (auto &f : frags) {
    const auto lw = getLen(f);
    lens.push_back(lw);
    f[1] = std::string(lw, '-');
  }

  for (uint64_t i = 0; i < nLines; ++i) {
    ost << "\n       ";
    for (uint64_t fi = 0; fi < frags.size(); ++fi) {
      ost << frags[fi][i] << getSpace(lens[fi], frags[fi][i]);
    }
  }
}

Proposals Graph::createProposalsAllInplace(const TensorIds &ids) {
  Proposals ps;
  ps.reserve(ids.size());
  for (auto id : ids) {
    ps.push_back({id, AliasType::allInplace()});
  }
  return ps;
}

std::ostream &operator<<(std::ostream &ost, const Graph &g) {
  g.append(ost);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const CrossAlias &ca) {
  ca.append(ost);
  return ost;
}

std::string Graph::typeString(OpId id) const { return op(id).typeString(); }

} // namespace inplace
} // namespace memory
} // namespace poprithms
