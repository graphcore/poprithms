// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "error.hpp"

#include <sstream>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/op.hpp>

namespace poprithms {
namespace common {
namespace compute {

void Op::insertInCopy(OutIndex outIndex, const CallEvent &ce) {
  auto &cs = inCopies_.at(outIndex.get());
  if (std::find(cs.cbegin(), cs.cend(), ce) == cs.cend()) {
    cs.push_back(ce);
  }
}

void Op::removeInCopy(OutIndex outIndex, const CallEvent &ce) {
  auto &cs   = inCopies_.at(outIndex.get());
  auto found = std::find(cs.cbegin(), cs.cend(), ce);
  if (found == cs.cend()) {
    std::ostringstream oss;
    oss << "Cannot remove call event ";
    ce.append(oss);
    oss << " from the set of in-copies at output index " << outIndex
        << " for this op, " << *this << ", as it is not present.";
    throw error(oss.str());
  }
  cs.erase(found);
}

void Op::insertOutCopy(OutIndex outIndex, const CallEvent &ce) {
  auto &cs = outCopies_.at(outIndex.get());
  if (std::find(cs.cbegin(), cs.cend(), ce) == cs.cend()) {
    cs.push_back(ce);
  }
}

void Op::removeOutCopy(OutIndex outIndex, const CallEvent &ce) {
  auto &cs   = outCopies_.at(outIndex.get());
  auto found = std::find(cs.cbegin(), cs.cend(), ce);
  if (found == cs.cend()) {
    std::ostringstream oss;
    oss << "Cannot remove call event ";
    ce.append(oss);
    oss << " from the set of out-copies at output index " << outIndex
        << " for this op, " << *this << ", as it is not present.";
    throw error(oss.str());
  }
  cs.erase(found);
}

Op::State Op::getComputeState() const {
  return {getSchedulableState(),
          outDTypes_,
          outDeviceIds_,
          inCopies_,
          outCopies_,
          *pGraph_};
}

bool Op::State::operator==(const Op::State &rhs) const {
  return outDTypes == rhs.outDTypes &&       //
         outDeviceIds == rhs.outDeviceIds && //
         inCopies == rhs.inCopies &&         //
         outCopies == rhs.outCopies &&       //
         pGraph == rhs.pGraph;               // pointer comparison.
}

Op::Op(const Op::State &ob)
    : schedulable::Op(ob.baseState), outDTypes_(ob.outDTypes),
      outDeviceIds_(ob.outDeviceIds), inCopies_(ob.inCopies),
      outCopies_(ob.outCopies), pGraph_(ob.pGraph) {
  if (!ob.pGraph) {
    throw error("Op's graph must not be nullptr");
  }
}

void Op::verifyInsSameDType() const {
  if (nInTensors() < 2) {
    return;
  }
  for (uint64_t i = 1; i < nInTensors(); ++i) {
    if (inDType(i) != inDType(0)) {
      std::ostringstream oss;
      oss << "Failure in verifyInsSameDType for op " << *this
          << ". The input #0 has dtype " << inDType(0) << ", but the input #"
          << i << " has dtype " << inDType(i) << '.';
      throw error(oss.str());
    }
  }
}

void Op::verifyOutsSameDType() const {
  if (nOutTensors() < 2) {
    return;
  }
  for (uint64_t i = 1; i < nOutTensors(); ++i) {
    if (outDType(i) != outDType(0)) {
      std::ostringstream oss;
      oss << "Failure in Op::verifyOutsSameDType for op " << *this
          << ". The output #0 has dtype " << outDType(0)
          << ", but the output #" << i << " has dtype " << outDType(i) << '.';
      throw error(oss.str());
    }
  }
}

void Op::verifyAllSameDType() const {

  verifyInsSameDType();
  verifyOutsSameDType();

  if (nInTensors() > 0 && nOutTensors() > 0) {
    if (outDType(0) != inDType(0)) {
      std::ostringstream oss;
      oss << "Failure in Op::verifyAllSameDType for op " << *this
          << ". The output #0 has dtype " << outDType(0)
          << ", but the input #0"
          << " has dtype " << inDType(0) << '.';
      throw error(oss.str());
    }
  }
}

DType Op::dtype(Port p, uint64_t i) const {
  return (p == Port::In ? inDType(InIndex(i)) : outDType(OutIndex(i)));
}

void Op::computeOpRemoveInputs(const ContiguousInIndexSubset &) {
  // nothing to do, as the op has no input specific attributes.
}

void Op::computeOpRemoveOutputs(const ContiguousOutIndexSubset &coin) {
  coin.reduce(outCopies_);
  coin.reduce(inCopies_);
  coin.reduce(outDTypes_);
  coin.reduce(outDeviceIds_);
}

DType Op::inDType(InIndex i) const { return graph().dtype(inTensorId(i)); }

DeviceIds Op::inDeviceIds() const {
  DeviceIds ins;
  ins.reserve(nInTensors());
  for (auto inId : inTensorIds()) {
    ins.push_back(graph().deviceId(inId));
  }
  return ins;
}

DeviceId Op::deviceId(Port p, uint64_t i) const {
  return (p == Port::In ? inDeviceId(InIndex(i)) : outDeviceId(OutIndex(i)));
}

DeviceId Op::inDeviceId(InIndex i) const {
  return graph().deviceId(inTensorId(i));
}

const Graph &Op::graph() const { return *pGraph_; }

TensorInfo Op::inTensorInfo(InIndex i) const {
  return TensorInfo(inShape(i), inDeviceId(i), inDType(i));
}

TensorInfos Op::inTensorInfos() const {
  std::vector<TensorInfo> infos;
  infos.reserve(nInTensors());
  for (uint64_t i = 0; i < nInTensors(); ++i) {
    infos.push_back(inTensorInfo(i));
  }
  return TensorInfos(std::move(infos));
}

TensorInfos Op::outTensorInfos() const {

  std::vector<TensorInfo> infos;
  infos.reserve(nOutTensors());
  for (uint64_t o = 0; o < nOutTensors(); ++o) {
    infos.push_back(outTensorInfo(o));
  }

  return TensorInfos(std::move(infos));
}

TensorInfo Op::outTensorInfo(OutIndex o) const {
  return TensorInfo(outShape(o), outDeviceId(o), outDType(o));
}

} // namespace compute
} // namespace common
} // namespace poprithms
