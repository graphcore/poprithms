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

void Graph::removeMultioutOp(OpId opId,
                             const OptionalTensorIds &substitutes,
                             const std::string &context) {

  const auto getBaseErr = [this, opId, &substitutes, &context]() {
    std::ostringstream oss;
    oss << "Failed to remove the op " << typeString(opId)
        << " in Graph::removeMultioutOp(opId = " << opId
        << ", substitutes = " << substitutes << ", context = " << context
        << "). ";
    return oss.str();
  };

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

  // reset the consumers of the op's output tensors with substitutes
  // (replace t0 with substitute0, etc.).
  for (uint64_t o = 0; o < nOutTensors(opId); ++o) {
    if (nConsumptionIds({opId, o}) != 0) {

      if (!substitutes.at(o).has_value()) {
        std::ostringstream oss;
        oss << getBaseErr() << ". Expected a replacement for output at index "
            << o << ", as there are consumer(s) ("
            << consumptionIds({opId, o}) << ") of this tensor.";
        throw error(oss.str());
      }
      const auto substitute = substitutes.at(o).value();

      if (shape(substitute) != shape({opId, o})) {
        std::ostringstream oss;
        oss << getBaseErr() << ". The shape of substitute at output index "
            << o << ", " << shape(substitute)
            << " is not the same as the shape of the substitutee "
            << shape({opId, o}) << '.';
        throw error(oss.str());
      }

      for (auto c : consumptionIds({opId, o})) {
        op(c.opId()).resetInTensor(c.inIndex(), substitute);
        op(substitute.opId()).insertConsumptionId(substitute.outIndex(), c);
      }
    }
  }

  // reset the producers of the inputs, as this is no longer a consumer.
  for (uint64_t i = 0; i < nInTensors(opId); ++i) {
    const auto inTensor = inTensorId(opId, InIndex());
    auto inOp           = inTensor.opId();
    op(inOp).removeConsumptionId(inTensor.outIndex(), ConsumptionId(opId, i));
  }

  // register removal event, and perform removal.
  removals.insert({opId, op(opId).getName(), ops_.size(), context});
  ops_[opId.get()].uptr.reset();
  live_.erase(opId);
}

void Graph::assertMultioutGraphCorrectness() const {
  {
    auto L = live_.size();
    auto R = removals.size();
    auto T = ops_.size();
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

  if (tId.opId().get() >= static_cast<int64_t>(ops_.size())) {
    std::ostringstream oss;
    oss << "Failure in confirmValidTensorId(TensorId=" << tId << "). "
        << "In total only " << ops_.size()
        << " Ops have ever been created in this Graph. ";
    throw error(oss.str());
  }

  if (!ops_[tId.opId().get()].uptr) {
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

  ops_.emplace_back(std::move(createdOp));

  const auto newId = ops_.back().uptr->id();

  live_.insert(/* hint = */ live_.end(), newId);

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
  return getName() == rhs.getName() && ops_ == rhs.ops_ &&
         live_ == rhs.live_ && typeid(*this) == typeid(rhs) &&
         multiOutTypeSpecificEqualTo(rhs) && removals == rhs.removals;
}

const Op &Graph::op(OpId a) const {

  const auto &opPtr = ops_[static_cast<uint64_t>(a.get())].uptr;

  if (!opPtr) {
    if (removals.registered(a)) {
      std::ostringstream oss;
      oss << "Invalid call, multiout::Graph::op(OpId = " << a << "). The op "
          << a << " was deleted, in RemovalEvent: \n     "
          << removals.event(a).str() << ". ";
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
  for (uint64_t i = 0; i < nOps(); ++i) {
    for (uint64_t o = 0; o < op(i).nOutTensors(); ++o) {
      ids.push_back({i, o});
    }
  }
  return ids;
}

OpIds Graph::opIds() const { return OpIds(live_.cbegin(), live_.cend()); }

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
  TensorIds outs       = outTensorIds(i);
  TensorIds insAndOuts = inTensorIds(i);
  insAndOuts.insert(insAndOuts.end(), outs.cbegin(), outs.cend());
  return insAndOuts;
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
