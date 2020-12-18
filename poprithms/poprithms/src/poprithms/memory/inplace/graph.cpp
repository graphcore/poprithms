// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "ops.hpp"

#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/schedule/scc/scc.hpp>
#include <poprithms/schedule/vanilla/vanilla.hpp>
#include <poprithms/util/printiter.hpp>
#include <util/copybyclone_impl.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

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

DisjointRegions Graph::outRegions(const DisjointRegions &inRegions,
                                  InIndex inIndex,
                                  OpId opId,
                                  OutIndex outIndex) const {
  return op(opId).outRegions(inRegions, inIndex, outIndex);
}

DisjointRegions Graph::inRegions(const DisjointRegions &out,
                                 InIndex inIndex,
                                 OpId opId,
                                 OutIndex outIndex) const {
  return op(opId).inRegions(out, inIndex, outIndex);
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

  auto paddings = (bp == BroadcastPadding::Yes)
                      ? createBroadcastPadElements(shape(id), l, u, cp)
                      : createNonAliasedPadElements(shape(id), l, u, cp);
  auto current = id;
  for (uint64_t d = 0; d < rank_u64(id); ++d) {
    current = concat(
        {std::get<0>(paddings[d]), current, std::get<1>(paddings[d])}, d);
  }
  return current;
}

TensorId Graph::slice(const TensorId &id, const Lower &l, const Upper &u) {
  return settSample(id, Region::fromBounds(shape(id), l, u));
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

template <class T, class... Args>
OpId Graph::createOp(const TensorIds &inIds,
                     const Shapes &outShapes,
                     Args... args) {
  return insertOp(
      std::make_unique<T>(
          Op::getBaseState(
              nOps_i64(), inIds, shapes(inIds), outShapes, opIds(inIds)),
          args...),
      inIds);
}

OpId Graph::insertOp(std::unique_ptr<Op> createdOp, const TensorIds &inIds) {

  scheduleIsValid = false;

  for (uint64_t inIndex = 0; inIndex < inIds.size(); ++inIndex) {
    const auto inTensor = inIds[inIndex];
    const auto sourceId = inTensor.opId();
    auto &source        = op(sourceId);
    source.insertConsumer(inTensor.outIndex(),
                          {createdOp->id(), InIndex(inIndex)});
  }

  ops.emplace_back(std::move(createdOp));

  ops.back().uptr->grow(aGraph(), tensorMap);
  auto newId = ops.back().uptr->id();

  return newId;
}

TensorId Graph::settSample(const TensorId &id, const Region &r) {
  return {createOp<SettSample>({id}, {r.nelms()}, r), 0};
}

TensorId
Graph::subSample(const TensorId &id, int64_t stride, uint64_t dimension) {
  return settSample(
      id, Region::fromStripe(shape(id), dimension, {1, stride - 1, 0}));
}

TensorId Graph::subSample(const TensorId &id, const Strides &strides) {
  std::vector<nest::Sett> setts;
  setts.reserve(strides.size());
  for (auto stride : strides.get()) {
    setts.push_back({{{1, stride - 1, 0}}});
  }
  return settSample(id, Region(shape(id), setts));
}

TensorId Graph::dimShuffle(const TensorId &id, const Permutation &perm) {
  return {createOp<DimShuffle>({id}, {shape(id).dimShuffle(perm)}, perm), 0};
}

TensorId Graph::reverse(const TensorId &id, const Dimensions &d) {
  return {createOp<Reverse>({id}, {shape(id)}, d.get()), 0};
}

TensorId Graph::flatten(const TensorId &id) {
  return reshape(id, {shape(id).nelms()});
}

TensorId Graph::reshape(const TensorId &id, const Shape &outShape) {
  return {createOp<Reshape>({id}, {outShape}), 0};
}

TensorId Graph::expand(const TensorId &id, const Shape &outShape) {
  shape(id).assertCanExpandTo(outShape);
  return {createOp<Expand>({id}, {outShape}), 0};
}

TensorId Graph::unary(const TensorId &id) {
  return {createOp<UnaryModifier>({id}, {shape(id)}), 0};
}

TensorId Graph::constant(const Shape &shape) {
  return {createOp<Alloc>({}, {shape}, ConstantColor), 0};
}

TensorId Graph::variable(const Shape &shape) {
  return {createOp<Alloc>({}, {shape}, VariableColor), 0};
}

TensorId Graph::concat(const TensorIds &ids, uint64_t axis) {
  if (ids.empty()) {
    std::ostringstream oss;
    oss << "In Tensor::concatIds(ids of size 0"
        << ", axis = " << axis << "). ids must be non-empty.";
    throw error(oss.str());
  }

  const auto opId =
      createOp<Concat>(ids, {Shape::concat(shapes(ids), axis)}, axis);

  return {opId, 0};
}

OpIds Graph::opIds(const TensorIds &tids) {
  OpIds ids_(tids.size());
  for (uint64_t i = 0; i < tids.size(); ++i) {
    ids_[i] = tids[i].opId();
  }
  return ids_;
}

void Graph::setName(const OpId id, const std::string &name) {
  op(id).setName(name);
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

  const auto &opPtr = ops[static_cast<uint64_t>(a.get())].uptr;

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

OpeningStatuses Graph::tryOpenings(const Proposals &proposals,
                                   CheckParallelWriteable check) {
  OpeningStatuses statuses;
  statuses.reserve(proposals.size());
  for (const auto &p : proposals) {
    statuses.push_back(tryOpening(p, check));
  }
  return statuses;
}

OpeningStatuses Graph::tryOpenings0(const TensorIds &ids,
                                    CheckParallelWriteable xp) {
  return tryOpenings(Proposal::open0(ids), xp);
}

OpeningStatuses Graph::tryOpenings0(const OpIds &ids,
                                    CheckParallelWriteable xp) {
  return tryOpenings(Proposal::open0(ids), xp);
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

Consumers Graph::consumers(const TensorId &id) const {
  return op(id.opId()).consumers(id.outIndex());
}

TensorIds Graph::difference(const TensorIds &a_, const TensorIds &b_) const {
  auto a = a_;
  auto b = b_;
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
// 1) Cache aliases between calls to tryOpening (T29079)
//
// 2) use the DAG structure to reduce alias computation, and sparsify
//    constraint calculation and insertion (T29080)

const Mux &Graph::asMux(OpId mid) const {
  auto proposedBaseOp   = &op(mid);
  auto proposedMuxOpPtr = dynamic_cast<const Mux *>(proposedBaseOp);
  if (!proposedMuxOpPtr) {
    std::ostringstream oss;
    oss << "Failure to cast Op to Mux. This for OpId = " << mid
        << ", where the Op trying to cast to Mux is " << *proposedBaseOp;
    throw error(oss.str());
  }
  auto &mux = *proposedMuxOpPtr;
  return mux;
}

InIndex Graph::muxInIndex(OpId mid) const { return asMux(mid).inIndex(); }

bool Graph::muxIsClosed(OpId mid) const { return asMux(mid).closed(); }

// See Scott Meyers' "Effective C++"
Mux &Graph::asMux(OpId mid) {
  return const_cast<Mux &>(static_cast<const Graph &>(*this).asMux(mid));
}

OpeningResult Graph::tryOpeningPartial(const Proposal &p,
                                       CheckParallelWriteable check) {

  auto &mux_ = asMux(p.muxId());

  if (mux_.nInTensors() <= p.inIndex().get()) {
    std::ostringstream oss;
    oss << "Invalid proposal input index, " << p.inIndex()
        << ", for mux with only " << mux_.nInTensors() << " input Tensors. ";
    throw error(oss.str());
  }

  const auto inTensorToAlias  = mux_.inTensorId(p.inIndex());
  const auto outTensorToAlias = mux_.outTensorId(0);

  if (!mux_.outplace()) {
    return OpeningResult::alreadyOpen();
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
    TensorId id;         // The Tensor whose info is stored
    Consumers modifiers; // modifiers of Tensor "id"
    TensorIds aliases;   // aliases of Tensor "id"
  };

  // Aliases of inputs which have modifiers, under the proposal.
  //
  // Example: proposal to open mux.
  //          input to mux is X
  //          Y is an alias of X, and Y has a modifier, so AliInfo of Y will
  //          be in the returned vector.
  //
  //            Y         .
  //           / \        .
  //      slice   unary   .
  //        |             .
  //        X             .
  //        |             .
  //       mux (proposal to make reshape_)
  //
  std::vector<AliInfo> inAliasModified;
  for (auto ali : allAliases(inTensorToAlias)) {
    const auto m = modifiers(ali);
    if (!m.empty()) {
      inAliasModified.push_back({ali, m, allAliases(ali)});
    }
  }

  // Aliases of outputs which have modifiers under the proposal.
  //
  // Example: proposal to open mux.
  //          output of slice is Y
  //          Z is an alias of Y, and Z has a modifier, so AliInfo of Z will
  //          be in the returned vector.
  //
  //     X
  //     |
  //    mux
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
  for (auto ali : allAliases(outTensorToAlias)) {
    const auto m = modifiers(ali);
    if (!m.empty()) {
      outAliasModified.push_back({ali, m, allAliases(ali)});
    }
  }

  // Open the Mux, let the aliases "flow"
  mux_.openAt(aGraph(), tensorMap, p.inIndex());

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

    // Check that no modifiers modify a Tensor with a constant or a self-alias
    // If they do, mux will remain close.
    for (auto m : allModifiers) {
      const auto &op_ = op(m.opId());
      const auto tId  = tensorMap.toAliasGraphId(op_.inTensorId(m.inIndex()));
      if (aGraph().containsColor(tId, ConstantColor) ||
          aGraph().containsAliases(tId)) {
        mux_.close(aGraph(), tensorMap);
        return OpeningResult::notParallelWriteable();
      }
    }
  }

  Constraints newConstraints;

  // The modifiers of output aliases might have new aliases: Tensors which are
  // aliased to the inputs of mux. Consumers of these new aliases must
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
  // aliased to outputs of the mux.  If the modifiers were scheduled
  // after mux, they must be scheduled after the new aliases too, so
  // that behaviour is unchanged.
  //
  // optimization here: don't need to go
  // through all in modifiers, only the ones which might be scheduled first
  // (T29080)
  for (const auto &x : inAliasModified) {
    auto diff = difference(allAliases(x.id), x.aliases);
    for (auto m : x.modifiers) {
      if (scheduleIndex(m.opId()) > scheduleIndex(mux_.id())) {

        // remoteinplace (popart) test requires this constraint TODO(T29862):
        // as follow-up, remove this.
        newConstraints.push_back({mux_.id(), m.opId()});

        for (auto newAlias : diff) {
          for (auto c : consumers(newAlias)) {
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
    constraints(newConstraints);
    scheduleIsValid = true;
    return OpeningResult::validWithUnchangedSchedule(
        std::move(newConstraints));
  }

  auto proposedSchedule =
      toOpIds(poprithms::schedule::vanilla::getSchedule_i64(
          getFwdEdges(newConstraints),
          schedule::vanilla::ErrorIfCycle::No,
          schedule::vanilla::VerifyEdges::No));

  if (proposedSchedule.size() != nOps()) {
    mux_.close(aGraph(), tensorMap);
    return OpeningResult::cycle();
  }

  return OpeningResult::validWithChangedSchedule(std::move(newConstraints),
                                                 std::move(proposedSchedule));
}

OpeningStatus Graph::tryOpening(const Proposal &p,
                                const CheckParallelWriteable check) {

  auto partial = tryOpeningPartial(p, check);
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
  asMux(proposal.muxId()).close(aGraph(), tensorMap);
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
      outIndex__[l]      = std::to_string(o);
      tensorId__[l]      = std::to_string(aliasId.get());
      tensorShape__[l]   = getStr(aGraph().shape(aliasId).get());
      tensorType__[l]    = aGraph().typeString(aliasId);
      selfAliases__[l]   = aGraph().containsAliases(aliasId) ? "yes" : "no";
      constants__[l] =
          aGraph().containsColor(aliasId, ConstantColor) ? "yes" : "no";
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

std::ostream &operator<<(std::ostream &ost, const Graph &g) {
  g.append(ost);
  return ost;
}

std::string Graph::typeString(OpId id) const { return op(id).typeString(); }

OpId Graph::multi(const TensorIds &inIds,
                  const Shapes &outShapes,
                  const CrossLinks &mapping) {
  return createOp<Multi>(inIds, outShapes, mapping);
}

TensorId Graph::mux(const TensorIds &ids) {
  return {createOp<Mux>(ids, {Shape::numpyVariadic(shapes(ids))}), 0};
}

TensorId Graph::mux(const TensorIds &ids, InIndex inInd) {
  if (inInd < 0) {
    return mux(ids);
  }
  return {createOp<Mux>(ids, {Shape::numpyVariadic(shapes(ids))}, inInd), 0};
}

} // namespace inplace
} // namespace memory

namespace util {
template class CopyByClone<memory::inplace::Op>;
}

} // namespace poprithms
