// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <memory>
#include <numeric>
#include <sstream>
#include <type_traits>

#include <common/multiout/error.hpp>

#include <poprithms/common/multiout/graph.hpp>
#include <poprithms/common/multiout/op.hpp>
#include <poprithms/util/contiguoussubset.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace common {
namespace multiout {

void Op::insertConsumptionId(OutIndex o, const ConsumptionId &c) {
  if (c.opId() == id()) {
    std::ostringstream oss;
    oss << "Invalid ConsumptionId " << c << " for of " << id() << "(" << *this
        << "). An op cannot consume its own output. ";
    throw error(oss.str());
  }

  auto &cs = consumptionIds_.at(o.get());
  if (std::find(cs.cbegin(), cs.cend(), c) == cs.cend()) {
    cs.push_back(c);
  }
}

InIndices Op::inIndices() const {
  InIndices inIndices_(nInTensors());
  for (uint64_t i = 0; i < nInTensors(); ++i) {
    inIndices_[i] = InIndex(i);
  }
  return inIndices_;
}

OutIndices Op::outIndices() const {
  OutIndices outIndices_(nOutTensors());
  for (uint64_t i = 0; i < nOutTensors(); ++i) {
    outIndices_[i] = OutIndex(i);
  }
  return outIndices_;
}

void Op::unimplemented(const std::string &n) const {
  std::ostringstream oss;
  oss << "This method for this class derived from multiout::Op "
      << "is not implemented. "
      << "Called on graph with name '" << getName()
      << "'. typeid of class (typeid(*this).name) is " << typeid(*this).name()
      << '.';
  if (!n.empty()) {
    oss << " Context: " << n;
  }
  throw error(oss.str());
}

TensorIds Op::inTensorIds(const InIndices &indices) const {
  TensorIds ids;
  ids.reserve(indices.size());
  for (auto i : indices) {
    ids.push_back(inTensorId(i));
  }
  return ids;
}

TensorIds Op::inTensorIdsExcluding(const InIndices &indices) const {
  TensorIds ids;
  for (InIndex i = 0; i < nInTensors(); ++i) {
    if (std::find(indices.cbegin(), indices.cend(), i) == indices.cend()) {
      ids.push_back(inTensorId(i));
    }
  }
  return ids;
}

namespace {

template <typename T>
void tVerifyDistinct(uint64_t N,
                     const std::vector<T> &vs,
                     const Op &op,
                     bool isIn) {

  auto low = [isIn]() { return isIn ? "in" : "out"; };
  auto upp = [isIn]() { return isIn ? "In" : "Out"; };

  std::vector<bool> removalMask(N, false);
  for (auto i : vs) {
    if (i.get() >= N) {
      std::ostringstream oss;
      oss << "Invalid " << upp() << "Index " << i << " for " << op
          << ", which only has " << N << " " << low() << "puts.";
      throw error(oss.str());
    }
    if (removalMask[i.get()]) {
      std::ostringstream oss;
      oss << "Duplicate " << upp() << "Index " << i
          << " in verifyDistinct for " << op << ", with " << upp()
          << "Indices ";
      util::append(oss, vs);
      oss << '.';
      throw error(oss.str());
    }
    removalMask[i.get()] = true;
  }
}
} // namespace

void Op::verifyDistinct(const InIndices &inIndices) const {
  tVerifyDistinct<InIndex>(nInTensors(), inIndices, *this, true);
}

void Op::verifyDistinct(const OutIndices &outIndices) const {
  tVerifyDistinct<OutIndex>(nOutTensors(), outIndices, *this, false);
}

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

TensorIds Op::outTensorIds(const OutIndices &os) const {
  TensorIds outIds;
  outIds.reserve(os.size());
  for (auto o : os) {
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

std::vector<uint64_t> Op::nConsumptionIds() const {
  std::vector<uint64_t> ns;
  ns.reserve(nOutTensors());
  for (OutIndex o = 0; o < nOutTensors(); ++o) {
    ns.push_back(nConsumptionIds(o));
  }
  return ns;
}

uint64_t Op::totalConsumptionIds() const {
  auto ns = nConsumptionIds();
  return std::accumulate(ns.cbegin(), ns.cend(), 0ull);
}

TensorIds Op::inAndOutTensorIds() const {
  TensorIds outs       = outTensorIds();
  TensorIds insAndOuts = inTensorIds();
  insAndOuts.insert(insAndOuts.end(), outs.cbegin(), outs.cend());
  return insAndOuts;
}

void Op::resetInTensorId(InIndex i, const TensorId &repl) {
  if (repl.opId() == id()) {
    std::ostringstream oss;
    oss << "Cannot replace input " << i << " of op " << id()
        << " with tensor " << repl << " as this is an output of op " << id()
        << '.';
    throw error(oss.str());
  }

  inIds_[i.get()] = repl;
}

Shape Op::shape(Port p, uint64_t i) const {
  return (p == Port::In ? inShape(InIndex(i)) : outShape(OutIndex(i)));
}

TensorId Op::tensorId(Port p, uint64_t i) const {
  return (p == Port::In ? inTensorId(InIndex(i)) : outTensorId(OutIndex(i)));
}

void Op::verifyRank(InIndex inIndex, uint64_t r) const {
  if (inShape(inIndex).rank_u64() != r) {
    std::ostringstream oss;
    oss << "Failure in verifyRank for op " << *this << ", at input index "
        << inIndex << ". Expected the rank to be " << r
        << ", but the tensor has shape " << inShape(inIndex) << " (rank "
        << inRank(inIndex) << ").";
    throw error(oss.str());
  }
}

uint64_t Op::nTensors(Port p) const {
  return (p == Port::In ? nInTensors() : nOutTensors());
}
std::string Op::lowercase(Port p) { return (p == Port::In ? "in" : "out"); }

void Op::verifyValidOutIndex(OutIndex index) const {
  if (index.get() >= nOutTensors()) {
    std::ostringstream oss;
    oss << "Invalid OutIndex (" << index << "). This Op " << *this
        << " only has " << nOutTensors() << " output Tensors. ";
    throw error(oss.str());
  }
}

void Op::verifyNInAndOutTensors(uint64_t expectedIn,
                                uint64_t expectedOut) const {
  verifyNTensors(Port::In, expectedIn);
  verifyNTensors(Port::Out, expectedOut);
}

void Op::verifyNTensors(Port p, uint64_t expected) const {
  if (nTensors(p) != expected) {
    std::ostringstream oss;
    oss << "Incorrect number of " << lowercase(p) << "put tensors, "
        << nTensors(p) << ", for op " << *this << ", expected " << expected
        << '.';
    throw error(oss.str());
  }
}

} // namespace multiout
} // namespace common
} // namespace poprithms
