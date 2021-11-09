// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <memory>
#include <numeric>
#include <sstream>
#include <type_traits>

#include <common/multiout/error.hpp>

#include <poprithms/common/multiout/graph.hpp>
#include <poprithms/common/multiout/op.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace common {
namespace multiout {

Shapes Op::inShapes() const {
  Shapes shapes;
  shapes.reserve(nInTensors());
  for (const auto &inTensorId : inTensorIds()) {
    shapes.push_back(multioutGraph().shape(inTensorId));
  }
  return shapes;
}

Shapes Op::State::inShapes() const {
  Shapes shapes;
  shapes.reserve(inIds.size());
  for (const auto &inTensorId : inIds) {
    shapes.push_back(multioutGraph.shape(inTensorId));
  }
  return shapes;
}

Shape Op::inShape(InIndex i) const {
  const auto s = multioutGraph().shape(inTensorId(i));
  return s;
}

uint64_t Op::inRank(InIndex i) const {
  return multioutGraph().rank_u64(inTensorId(i));
}

uint64_t Op::nInElms(InIndex i) const {
  return multioutGraph().nelms(inTensorId(i));
}

Shape Op::State::inShape(InIndex i) const {
  return multioutGraph.shape(inIds[i.get()]);
}

void Op::setGraph(const Graph &g_) { multioutGraph_ = &g_; }

std::vector<OutIndex> Op::outIndicesConsumed() const {
  std::vector<OutIndex> os;
  for (uint64_t o = 0; o < nOutTensors(); ++o) {
    if (!consumptionIds(o).empty()) {
      os.push_back(o);
    }
  }
  return os;
}

Op::State::State(const OpId id_,
                 const TensorIds &inIds_,
                 const std::vector<ConsumptionIds> &consumptionIds_,
                 const Shapes &outShapes_,
                 const std::string &name_,
                 const Graph &multioutGraph_)
    : id(id_), inIds(inIds_), consumptionIds(consumptionIds_),
      outShapes(outShapes_), name(name_), multioutGraph(multioutGraph_) {

  if (consumptionIds.size() != outShapes.size()) {
    std::ostringstream oss;
    oss << "The size of the vectors consumptionIds and outShapes "
        << "should be the same in the multiout::Op::State constructor. "
        << "But consumptionIds is of size " << consumptionIds.size()
        << ", and outShapes is of size " << outShapes.size() << ". "
        << "They should both be exactly equal to the number of outputs. ";
    throw error(oss.str());
  }
}

Op::~Op() = default;

Op::Op(const State &ob)
    : id_(ob.id), inIds_(ob.inIds), consumptionIds_(ob.consumptionIds),
      outShapes_(ob.outShapes), name_(ob.name),
      multioutGraph_(&ob.multioutGraph) {}

// C++20 we will be able to just use (= default).
bool Op::State::operator==(const State &rhs) const {
  return                                      //
      id == rhs.id &&                         //
      inIds == rhs.inIds &&                   //
      consumptionIds == rhs.consumptionIds && //
      outShapes == rhs.outShapes &&           //
      name == rhs.name;
  // we do not compare graphs.
}

TensorIds Op::outTensorIds() const {
  TensorIds outIds;
  outIds.reserve(nOutTensors());
  for (uint64_t o = 0; o < nOutTensors(); ++o) {
    outIds.push_back(TensorId{id(), OutIndex(o)});
  }
  return outIds;
}

std::ostream &operator<<(std::ostream &os, const Op &op) {
  os << op.str();
  if (!op.getName().empty()) {
    os << "::" << op.getName();
  }
  return os;
}

void Op::verify(InIndex inIndex,
                OutIndex outIndex,
                const std::string &context) const {
  if (inIndex >= nInTensors()) {
    std::ostringstream oss;
    oss << "Invalid InIndex in " << context << " of " << typeString()
        << ". InIndex=" << inIndex << ", but nInTensors=" << nInTensors()
        << '.';
    throw error(oss.str());
  }
  if (outIndex >= nOutTensors()) {
    std::ostringstream oss;
    oss << "Invalid OutIndex in " << context << " of " << typeString()
        << ". InIndex=" << inIndex << ", but nOutTensors=" << nOutTensors()
        << '.';
    throw error(oss.str());
  }
}

std::string Op::str() const {
  return typeString() + std::string("::") + id();
}

bool Op::isConsumptionId(OutIndex o, const ConsumptionId &cid) const {
  const auto &cs = consumptionIds_.at(o.get());
  return std::find(cs.begin(), cs.end(), cid) != cs.cend();
}

bool Op::operator==(const Op &rhs) const {
  return
      // Same common properties:
      getState() == rhs.getState() &&
      // Same derived class:
      typeid(*this) == typeid(rhs) &&
      // Same derived class properties:
      multiOutTypeSpecificEqualTo(rhs);
}

void Op::removeConsumptionId(OutIndex o, const ConsumptionId &toRemove) {
  auto &ids  = consumptionIds_[o.get()];
  auto found = std::find(ids.begin(), ids.end(), toRemove);
  if (found == ids.cend()) {
    std::ostringstream oss;
    oss << "Attempting to remove ConsumptionId " << toRemove
        << " from this op, " << id() << " at output index " << o
        << ". But ConsumptionId not present. "
        << "The ConsumptionIds are " << ids << ". ";
    throw error(oss.str());
  }
  ids.erase(found);
}

TensorIds Op::inAndOutTensorIds() const {
  TensorIds outs       = outTensorIds();
  TensorIds insAndOuts = inTensorIds();
  insAndOuts.insert(insAndOuts.end(), outs.cbegin(), outs.cend());
  return insAndOuts;
}

} // namespace multiout
} // namespace common
} // namespace poprithms
