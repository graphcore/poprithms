// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

#include <common/multiout/error.hpp>

#include <poprithms/common/multiout/graph.hpp>
#include <poprithms/common/multiout/op.hpp>
#include <poprithms/util/copybyclone_impl.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace common {
namespace multiout {

FwdEdgeMap Graph::getMultioutForwardEdgeMap_u64() const {

  auto outOps = [this](OpId opId) {
    OpIds outs;
    auto outTensors = outTensorIds(opId);
    for (auto t : outTensorIds(opId)) {
      for (auto c : consumptionIds(t)) {
        outs.push_back(c.opId());
      }
    }
    return outs;
  };

  FwdEdgeMap fwdEdgeMap(opIds());

  for (auto id : opIds()) {
    const auto outs = outOps(id);
    fwdEdgeMap.reserve(id, outs.size());
    for (auto out : outs) {
      fwdEdgeMap.insertEdge(id, out);
    }
  }

  return fwdEdgeMap;
}

void Graph::resetGraphOfOps() {
  for (auto &op_ : ops()) {
    if (op_.uptr) {
      op_.uptr->setGraph(*this);
    }
  }
}

Graph::Graph(Graph &&g) {
  atts = std::move(g.atts);
  resetGraphOfOps();
}

Graph &Graph::operator=(Graph &&g) {
  if (this != &g) {
    atts = std::move(g.atts);
    resetGraphOfOps();
  }
  return *this;
}

Graph::Graph(const Graph &g) {
  atts = g.atts;
  resetGraphOfOps();
}

Graph &Graph::operator=(const Graph &g) {
  if (this != &g) {
    atts = g.atts;
    resetGraphOfOps();
  }
  return *this;
}

void Graph::verifyOpsConnectedToThisGraph() const {
  for (const auto &op_ : ops()) {
    if (op_.uptr) {
      if (&op_.uptr->multioutGraph() != this) {
        std::ostringstream oss;
        oss << "Failed to verify that ops of this graph (" << getName()
            << ") are connected to it. "
            << "The op " << *op_.uptr << " (one of the " << ops().size()
            << " ops created in this graph) "
            << "has graph pointer " << &op_.uptr->multioutGraph()
            << ", but this graph has address " << this << '.';
        throw error(oss.str());
      }
    }
  }
}

std::vector<OutIndex> Graph::outIndicesConsumed(OpId id) const {
  return op(id).outIndicesConsumed();
}

void Graph::verifyValidOutputSubstitute(const TensorId &before,
                                        const TensorId &after) const {

  multiOutTypeSpecificVerifyValidOutputSubstitute(before, after);
  if (shape(before) != shape(after)) {
    std::ostringstream oss;
    oss << "Failure in multiout::Graph::verifyValidOutputSubstitute, where "
        << "Shape before substitution is " << shape(before)
        << " and Shape after substitution is " << shape(after) << ". "
        << "Replacement (substitute) tensors must have the same Shape "
        << "as the tensors being replaced. "
        << "The creator of " << before << " is " << op(before.opId())
        << " and the creator of " << after << " is " << op(after.opId())
        << '.';
    throw error(oss.str());
  }
}

void Graph::verifyValidOutputSubstitutes(
    const OpId opId,
    const OptionalTensorIds &substitutes) const {

  for (auto outIndex : outIndicesConsumed(opId)) {
    const auto optionalSubstitute = substitutes.at(outIndex.get());
    if (!optionalSubstitute.has_value()) {
      std::ostringstream oss;
      oss << "There is no replacement for output at index " << outIndex
          << " of the op to remove, " << op(opId) << ". But "
          << "this output has consumers (" << consumptionIds({opId, outIndex})
          << "), and so a replacment must be provided. ";
      throw error(oss.str());
    }
    verifyValidOutputSubstitute({opId, outIndex}, optionalSubstitute.value());
  }
}

void Graph::removeOp(OpId opId,
                     const OptionalTensorIds &substitutes,
                     const std::string &context) {

  verifyValidOutputSubstitutes(opId, substitutes);

  multiOutTypeSpecificRemoveOp(opId, substitutes);

  // rewire, in preparation for removing opId from the Graph.
  //
  //   >-------+           +------> t0 / substitute0 -+--- consumer
  //           |           |                          |
  //           |           |                          +--- consumer
  //           +---opId----+
  //           |           |
  //           |           |
  //   >-------+           +------> t1 / substitute1 ----- consumer
  //
  //

  // 1) when opId is removed, it no longer consumes its inputs. Register this
  //    fact with the producers of opId's inputs.
  for (uint64_t inIndex = 0; inIndex < nInTensors(opId); ++inIndex) {
    const auto inTensor = inTensorId(opId, inIndex);
    const auto inOp     = inTensor.opId();
    const auto outIndex = inTensor.outIndex();
    op(inOp).removeConsumptionId(outIndex, ConsumptionId(opId, inIndex));
  }

  // 2) wire up the substitutes. That is, all consumers of outputs of opId
  //    must change to consume the corresponding substitue.
  for (auto outIndex : outIndicesConsumed(opId)) {
    const TensorId substitute = substitutes.at(outIndex.get()).value();
    for (auto c : consumptionIds({opId, outIndex})) {
      op(c.opId()).resetInTensor(c.inIndex(), substitute);
      op(substitute.opId()).insertConsumptionId(substitute.outIndex(), c);
    }
  }

  // finally, register removal event, and perform removal.
  atts.removals_.insert({opId, op(opId).getName(), ops().size(), context});
  ops()[opId.get()].uptr.reset();
  atts.live_.erase(opId);
}

void Graph::assertMultioutGraphCorrectness() const {
  {
    auto L = atts.live_.size();
    auto R = atts.removals_.size();
    auto T = ops().size();
    if (L + R != T) {
      std::ostringstream oss;
      oss << "#live = " << L << " #removed = " << R
          << " total created = " << T << ". But " << L << " + " << R
          << " != " << T << ". Failed to assert correctness. ";
    }
  }

  // for every, for every input, check that there's agreement with the
  // producer of the input:
  for (auto op_ : opIds()) {
    for (uint64_t i = 0; i < nInTensors(op_); ++i) {
      auto in_ = inTensorId(op_, i);
      if (!op(in_.opId())
               .isConsumptionId(in_.outIndex(), ConsumptionId(op_, i))) {
        std::ostringstream oss;
        oss << "Op " << op_ << " has " << in_
            << " as an input, but is not registered as a consumer of it. ";
        throw error(oss.str());
      }

      if (op(in_.opId()).outShape(in_.outIndex()) != op(op_).inShape(i)) {
        std::ostringstream oss;
        oss << "output shape of " << in_.opId() << " at index "
            << in_.outIndex() << " does agree with input shape of " << op_
            << " at index. " << i;
        throw error(oss.str());
      }
    }
  }
}

uint64_t Graph::nConsumptionIds(const TensorId &id) const {
  return op(id.opId()).consumptionIds(id.outIndex()).size();
}

TensorId Graph::outTensorId(const OpTraversal &o) {
  return {o.opId(), o.outIndex()};
}

TensorId Graph::inTensorId(const OpTraversal &ot) const {
  return op(ot.opId()).inTensorId(ot.inIndex());
}

Shapes Graph::inShapes(OpId id) const { return op(id).inShapes(); }

Shapes Graph::outShapes(OpId id) const { return op(id).outShapes(); }

void Graph::confirmValidTensorId(const TensorId &tId) const {

  if (tId.opId().get() >= static_cast<int64_t>(ops().size())) {
    std::ostringstream oss;
    oss << "Failure in confirmValidTensorId(TensorId=" << tId << "). "
        << "In total only " << ops().size()
        << " Ops have ever been created in this Graph. ";
    throw error(oss.str());
  }

  if (!ops()[tId.opId().get()].uptr) {
    std::ostringstream oss;
    oss << "Failure in confirmValidTensorId(TensorId=" << tId << "). "
        << "The Op " << tId.opId() << " no longer exists.";
    throw error(oss.str());
  }

  if (tId.outIndex().get() >= op(tId.opId()).nOutTensors()) {
    std::ostringstream oss;
    oss << "Failure in confirmValidTensorId(TensorId=" << tId
        << "). Invalid output index (" << tId.outIndex() << ") as Op "
        << tId.opId() << "(" << op(tId.opId()) << ") only has "
        << op(tId.opId()).nOutTensors() << " output Tensors. ";
    throw error(oss.str());
  }
}

uint64_t Graph::nInTensors(OpId id) const { return op(id).nInTensors(); }
uint64_t Graph::nOutTensors(OpId id) const { return op(id).nOutTensors(); }

OpId Graph::insertMultioutOp(std::unique_ptr<Op> createdOp) {

  if (createdOp->id() != nxtOpId()) {
    std::ostringstream oss;
    oss << "Error in multiout::Graph::insertMultioutOp(createdOp="
        << *createdOp << "). "
        << "Expected createdOp to have OpId=" << nxtOpId()
        << ", the next new OpId available. ";
    throw error(oss.str());
  }

  for (uint64_t inIndex = 0; inIndex < createdOp->nInTensors(); ++inIndex) {
    const auto inTensor = createdOp->inTensorId(inIndex);
    const auto sourceId = inTensor.opId();
    auto &source        = op(sourceId);
    source.insertConsumptionId(inTensor.outIndex(),
                               {createdOp->id(), InIndex(inIndex)});
  }

  ops().emplace_back(std::move(createdOp));

  const auto newId = ops().back().uptr->id();

  atts.live_.insert(/* hint = */ atts.live_.end(), newId);

  return newId;
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

const std::string &Graph::getName(OpId opId) const {
  return op(opId).getName();
}

bool Graph::operator==(const Graph &rhs) const {
  return typeid(*this) == typeid(rhs) && atts == rhs.atts &&
         multiOutTypeSpecificEqualTo(rhs);
}

bool Graph::Attributes::operator==(const Graph::Attributes &rhs) const {
  return name_ == rhs.name_ && removals_ == rhs.removals_ &&
         live_ == rhs.live_ && ops_ == rhs.ops_;
}
const Op &Graph::op(OpId a) const {

  const auto &opPtr = ops()[static_cast<uint64_t>(a.get())].uptr;

  if (!opPtr) {
    if (atts.removals_.registered(a)) {
      std::ostringstream oss;
      oss << "Invalid call, multiout::Graph::op(OpId = " << a << "). The op "
          << a << " was deleted, in RemovalEvent: \n     "
          << atts.removals_.event(a).str() << ". ";
      throw error(oss.str());
    }

    // this should never happen. Ops should either be live, or have a removal
    // event registered for them.
    {
      std::ostringstream oss;
      oss << "nullptr in op(" << std::to_string(a.get()) << "). ";
      oss << "This is strange, Ops should either be live, or have a removal "
          << "event registered. Neither is true for this op. ";
      throw error(oss.str());
    }
  }
  return *opPtr;
}

// See Scott Meyers' "Effective C++"
Op &Graph::op(OpId id) {
  return const_cast<Op &>(static_cast<const Graph &>(*this).op(id));
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

ConsumptionIds Graph::consumptionIds(const TensorId &id) const {
  const auto cs = op(id.opId()).consumptionIds(id.outIndex());
  return cs;
}

TensorIds Graph::setDifference(const TensorIds &a_, const TensorIds &b_) {
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

uint64_t Graph::nOutTensors(const OpIds &ids) const {
  uint64_t n = 0;
  for (auto i : ids) {
    n += op(i).nOutTensors();
  }
  return n;
}

uint64_t Graph::nMultioutRows(const OpIds &opIds) const {
  return nOutTensors(opIds) + nWithZeroOutputs(opIds);
}

std::vector<std::string> Graph::getOpNames() const {
  std::vector<std::string> names;
  names.reserve(nOps());
  for (uint64_t i = 0; i < nOps(); ++i) {
    names.push_back(op(i).getName());
  }
  return names;
}

std::string Graph::typeString(OpId id) const { return op(id).typeString(); }

void Graph::verifyTensorId(const TensorId &tId) const {
  if (tId.opId() >= nOps()) {
    std::ostringstream oss;
    oss << "Failure in verifyTensorId(TensorId=" << tId
        << "), in this Graph, which only has " << nOps() << " Ops. "
        << "This TensorId has (creator) OpId " << tId.opId()
        << ", which was expected to be less than " << nOps() << ". ";
    throw error(oss.str());
  }

  const auto nOuts = op(tId.opId()).nOutTensors();
  if (tId.outIndex() >= nOuts) {
    std::ostringstream oss;
    oss << "Failure in verifyTensorId(TensorId=" << tId
        << "), as the creator Op, with OpId=" << tId.opId() << " only has "
        << nOuts << " outputs. ";
    throw error(oss.str());
  }
}

TensorIds Graph::tensorIds() const {
  TensorIds ids;
  ids.reserve(nTensors());
  for (auto opId : opIds()) {
    for (uint64_t o = 0; o < op(opId).nOutTensors(); ++o) {
      ids.push_back({opId, o});
    }
  }
  return ids;
}

TensorIds Graph::outTensorIds(OpId id) const { return op(id).outTensorIds(); }

TensorIds Graph::inTensorIds(OpId id) const { return op(id).inTensorIds(); }

TensorId Graph::inTensorId(OpId opId, InIndex inIndex) const {
  return op(opId).inTensorId(inIndex);
}

void Graph::setName(const TensorId &id, const std::string &name) {
  if (op(id.opId()).nOutTensors() != 1) {
    std::ostringstream oss;
    oss << "Cannot call Graph::setName(TensorId=" << id << ", name=" << name
        << "), because the Op creator has multiple outputs. "
        << "Call "
        << "setName(OpId=" << id.opId() << ", name=" << name << ") instead.";
    throw error(oss.str());
  }
  setName(id.opId(), name);
}

TensorIds Graph::inAndOutTensorIds(OpId i) const {
  return op(i).inAndOutTensorIds();
}

uint64_t Graph::nOpsWithZeroOutputs() const {
  return nWithZeroOutputs(opIds());
}

uint64_t Graph::nWithZeroOutputs(const OpIds &opIds) const {
  uint64_t N{0};
  for (auto i : opIds) {
    if (op(i).nOutTensors() == 0) {
      ++N;
    }
  }
  return N;
}

std::vector<util::StringColumn> Graph::getMultioutColumns() const {
  return getMultioutColumns(opIds());
}

std::vector<util::StringColumn>
Graph::getMultioutColumns(const OpIds &opIds) const {
  const auto nTens = nMultioutRows(opIds);

  using Strings = std::vector<std::string>;
  Strings opId(nTens, "");
  Strings name(nTens, "");
  Strings opType(nTens, "");
  Strings inTensors(nTens, "");
  Strings outIndex(nTens, "");
  Strings tensorShape(nTens, "");

  uint64_t ti = 0;
  for (auto i : opIds) {
    opId[ti]      = std::to_string(i.get());
    opType[ti]    = op(i).typeString();
    name[ti]      = op(i).getName();
    inTensors[ti] = util::getStr((op(i).inTensorIds()));
    for (uint64_t o = 0; o < op(i).nOutTensors(); ++o) {
      outIndex[ti]    = std::to_string(o);
      tensorShape[ti] = util::getStr(shape({i, o}).get());
      ++ti;
    }
    if (op(i).nOutTensors() == 0) {
      ++ti;
    }
  }

  std::vector<util::StringColumn> cols;
  cols.push_back({"OpId", opId});

  // only add debug strings if at least one has:
  if (std::any_of(name.cbegin(), name.cend(), [](const auto &x) {
        return !x.empty();
      })) {
    cols.push_back({"Name", name});
  }

  cols.push_back({"OpType", opType});
  cols.push_back({"InTensors", inTensors});

  if (std::any_of(outIndex.cbegin(), outIndex.cend(), [](const auto &x) {
        return (!x.empty() && x[0] != '0');
      })) {
    cols.push_back({"OutIndex", outIndex});
  }

  cols.push_back({"Shape", tensorShape});

  return cols;
}

} // namespace multiout
} // namespace common

namespace util {
template class CopyByClone<common::multiout::Op>;
}

} // namespace poprithms
