// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <sstream>

#include <common/compute/error.hpp>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/op.hpp>

namespace poprithms {
namespace common {
namespace compute {

void Op::initSimOut(SimTensorMap &stm) const {
  initializeSimOut(stm);

  // Some tensors might have pinned initial values:
  for (OutIndex o = 0; o < nOutTensors(); ++o) {
    if (outDeviceType(o) == DeviceType::Ipu) {
      for (auto &&[repl, val] : initialValues(o)) {
        stm.getValue({id(), o}).at(repl).update_(val);
      }
    }
  }
}

bool Op::hasDerivedRefs() const {
  for (OutIndex o = 0; o < nOutTensors(); ++o) {
    if (hasDerivedRefs(o)) {
      return true;
    }
  }
  return false;
}

bool Op::atLeastOneOutIsIpu() const {
  for (uint64_t outIndex = 0; outIndex < nOutTensors(); ++outIndex) {
    if (outDeviceType(outIndex) == DeviceType::Ipu) {
      return true;
    }
  }
  return false;
}

bool Op::isPartiallyHost() const {
  const auto types = inAndOutDeviceTypes();
  if (types.empty()) {
    return false;
  }

  // They're all host?
  if (std::all_of(types.cbegin(), types.cend(), [](auto t) {
        return t == DeviceType::Host;
      })) {
    return false;
  }

  // They're all not host?
  if (std::all_of(types.cbegin(), types.cend(), [](auto t) {
        return t != DeviceType::Host;
      })) {
    return false;
  }
  return true;
}

bool Op::inIsFixedPoint(InIndex inIndex) const {
  return poprithms::ndarray::isFixedPoint(inDType(inIndex));
}

bool Op::outIsFixedPoint(OutIndex outIndex) const {
  return poprithms::ndarray::isFixedPoint(outDType(outIndex));
}

Op::State Op::State::getStartingState(OpId opId,
                                      SubGraphId sgId,
                                      const TensorIds &ins,
                                      const TensorInfos &outs,
                                      const Graph &g) {
  return State(schedulable::Op::State::getStartingState(
                   opId, sgId, ins, outs.shapes(), g),
               outs.dtypes(),
               outs.deviceIds(),
               std::vector<CallEvents>(outs.size()),
               std::vector<CallEvents>(outs.size()),
               InitialValues(outs.size()),
               std::vector<TensorIds>(outs.size()));
}

template <typename T>
void checkNumberOfOuts(const T &t, const Op &op, const std::string &ctxt) {
  if (t.size() != op.nOutTensors()) {
    std::ostringstream oss;
    oss << "There are " << t.size() << " values (" << ctxt
        << ") for tensors but " << op.nOutTensors() << " output tensors. "
        << t.size() << " != " << op.nOutTensors() << ". This for op " << op;
    throw error(oss.str());
  }
}

InIndices Op::inputsWithDeviceType(DeviceType dt) const {
  InIndices onDevIns;
  for (InIndex i = 0; i < nInTensors(); ++i) {
    if (inDeviceType(i) == dt) {
      onDevIns.push_back(i);
    }
  }
  return onDevIns;
}

OutIndices Op::outputsWithDeviceType(DeviceType dt) const {
  OutIndices onDevOuts;
  for (OutIndex o = 0; o < nOutTensors(); ++o) {
    if (outDeviceType(o) == dt) {
      onDevOuts.push_back(o);
    }
  }
  return onDevOuts;
}

void Op::verifyValidAtComputeLevel() const {

  // check that there are right number of attributes (1 per output tensor).
  checkNumberOfOuts(outDTypes_, *this, "dtypes");
  checkNumberOfOuts(outDeviceIds_, *this, "device ids");
  checkNumberOfOuts(inCopies_, *this, "in copies");
  checkNumberOfOuts(inCopies_, *this, "out copies");
  checkNumberOfOuts(inCopies_, *this, "initial values");
  checkNumberOfOuts(inCopies_, *this, "derived refs");

  // Check that callee-copy relationships are correct:
  for (OutIndex o = 0; o < nOutTensors(); ++o) {
    for (const auto &ce : inCopies_.at(o.get())) {
      const auto &caller = graph().computeOp(ce.caller());
      // get the index copied in at.  This serves as the test.
      const auto index = caller.inIndex({outTensorId(o), ce.index()});
      (void)index;
    }

    for (const auto &ce : outCopies_.at(o.get())) {
      const auto &caller = graph().computeOp(ce.caller());
      // get the index copied out at.  This serves as the test.
      const auto index = caller.outIndex({outTensorId(o), ce.index()});
      (void)index;
    }
  }

  for (OutIndex o = 0; o < nOutTensors(); ++o) {
    for (const auto &[k, v] : initVals_.getInitialValues(o)) {
      (void)k;
      if (v.shape() != outShape(o) || v.dtype() != outDType(o)) {
        std::ostringstream oss;
        oss << "The initial value for the output #" << o << " has shape "
            << v.shape() << " and type " << v.dtype()
            << ", but the output tensor #" << o << " has shape "
            << outShape(o) << " and type " << outDType(o);
        throw error(oss.str());
      }
    }
  }

  for (OutIndex o = 0; o < nOutTensors(); ++o) {
    for (auto derivedRef : derivedRefs_.at(o.get())) {
      auto reported =
          graph().computeOp(derivedRef.opId()).rootRef(derivedRef.outIndex());
      if (reported != outTensorId(o)) {

        std::ostringstream oss;
        oss << "The output #" << o << " has " << derivedRef
            << " as a derived reference, but " << derivedRef << " has "
            << reported << " as its root reference. ";
        throw error(oss.str());
      }
    }
  }

  InIndices remIns   = inputsWithDeviceType(DeviceType::Remote);
  OutIndices remOuts = outputsWithDeviceType(DeviceType::Remote);

  if (!supportsRemote(remIns, remOuts)) {
    std::ostringstream oss;
    oss << "This op " << *this
        << " does not support this combination of remote tensor "
           "input/outputs. Remote inputs at ";
    util::append(oss, remIns);
    oss << " and remote outputs at ";
    util::append(oss, remOuts);
    oss << ". Note that only a handful of ops support remote tensor inputs "
           "and outputs.";
    throw error(oss.str());
  }
}

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
          initVals_,
          derivedRefs_};
}

bool Op::State::operator==(const Op::State &rhs) const {
  return outDTypes == rhs.outDTypes &&       //
         outDeviceIds == rhs.outDeviceIds && //
         inCopies == rhs.inCopies &&         //
         outCopies == rhs.outCopies &&       //
         initVals == rhs.initVals &&         //
         derivedRefs == rhs.derivedRefs;
}

Op::Op(const Op::State &ob)
    : schedulable::Op(ob.baseState), outDTypes_(ob.outDTypes),
      outDeviceIds_(ob.outDeviceIds), inCopies_(ob.inCopies),
      outCopies_(ob.outCopies), initVals_(ob.initVals),
      derivedRefs_(ob.derivedRefs) {}

DType Op::dtype(Port p, uint64_t i) const {
  return (p == Port::In ? inDType(InIndex(i)) : outDType(OutIndex(i)));
}

void Op::computeOpRemoveInputs(const ContiguousInIndexSubset &coin) {
  // nothing to do, as the op has no input specific attributes.

  computeDerivedRemoveInputs(coin);
}

void Op::computeOpRemoveOutputs(const ContiguousOutIndexSubset &coin) {
  coin.reduce(outCopies_);
  coin.reduce(inCopies_);
  coin.reduce(outDTypes_);
  coin.reduce(outDeviceIds_);
  initVals_.reduce(coin);

  // TODO(T26307): update the root stored in each of the derived refs.
  coin.reduce(derivedRefs_);
  computeDerivedRemoveOutputs(coin);
}

std::map<uint64_t, HostTensor> Op::initialValues(OutIndex o) const {

  if (!isIpu(o)) {
    throw error(
        "Only ipu tensors can have initial values before compilation. ");
  }

  return initVals_.getInitialValues(o);
}

void Op::setInitialValue(uint64_t replica, OutIndex o, const HostTensor &v) {

  if (!isIpu(o)) {
    throw error("Only ipu tensors can have initial values set, other tensors "
                "must be set after compilation. ");
  }

  if (v.shape() != outShape(o)) {
    std::ostringstream oss;
    oss << "The shape of the host tensor is " << v.shape()
        << ", which is different to the shape of output #" << o << " of op "
        << *this << ": " << outShape(o)
        << ". These 2 shapes must match in setInitialValue. ";
    throw error(oss.str());
  }

  if (v.dtype() != v.dtype()) {
    std::ostringstream oss;
    oss << "The data type of the host tensor is " << v.dtype()
        << ", which is different to the data type of output #" << o
        << " of op " << *this
        << ": data types must match in setInitialValue. ";
    throw error(oss.str());
  }

  if (replica >= graph().replicationFactor_u64()) {
    std::ostringstream oss;
    oss << "Attempt to set the initial value replica #" << replica
        << " of a graph with only " << graph().replicationFactor_u64() << ".";
    throw error(oss.str());
  }

  initVals_.setValue(o, replica, v);
}

DeviceType Op::outDeviceType(OutIndex o) const {
  return graph().device(outDeviceId(o)).deviceType();
  // This causes problems if called during construction:
  // return m().deviceType(outTensorId(o));
}

DeviceType Op::inDeviceType(InIndex i) const {
  return graph().deviceType(inTensorId(i));
}
DeviceType Op::deviceType(Port p, uint64_t i) const {
  return (p == Port::In ? inDeviceType(InIndex(i))
                        : outDeviceType(OutIndex(i)));
}

DeviceTypes Op::outDeviceTypes() const {
  std::vector<DeviceType> ts;
  ts.reserve(nOutTensors());
  for (uint64_t o = 0; o < nOutTensors(); ++o) {
    ts.push_back(outDeviceType(o));
  }
  return ts;
}

DeviceTypes Op::inDeviceTypes() const {
  std::vector<DeviceType> ts;
  ts.reserve(nInTensors());
  for (uint64_t i = 0; i < nInTensors(); ++i) {
    ts.push_back(inDeviceType(i));
  }
  return ts;
}

const Device &Op::outDevice(OutIndex o) const {
  return graph().device(outDeviceId(o));
}
const Device &Op::inDevice(InIndex i) const {
  return graph().device(inDeviceId(i));
}

DeviceTypes Op::inAndOutDeviceTypes() const {
  auto all        = inDeviceTypes();
  const auto outs = outDeviceTypes();
  all.insert(all.end(), outs.cbegin(), outs.cend());
  return all;
}

const Device &Op::device(Port p, uint64_t i) const {
  return (p == Port::In ? inDevice(InIndex(i)) : outDevice(OutIndex(i)));
}

DeviceType Op::deviceTypeByUnanimity() const {
  const auto ts = inAndOutDeviceTypes();
  if (ts.empty()) {
    std::ostringstream oss;
    oss << "Cannot infer device type for Op " << *this
        << " which has no inputs and no outputs. ";
    throw error(oss.str());
  }
  if (!std::all_of(
          ts.cbegin(), ts.cend(), [&ts](auto t) { return t == ts[0]; })) {
    std::ostringstream oss;
    oss << "Not all inputs and outputs for op " << *this
        << " are on a device of the same type. "
        << "It is therefore not possible to define what device type " << *this
        << "is on. "
        << "The inputs and outputs, " << inAndOutTensorIds()
        << " are on device types, ";
    poprithms::util::append(oss, ts);
    throw error(oss.str());
  }
  return ts[0];
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

const Graph &Op::computeGraph() const {
  return static_cast<const Graph &>(multioutGraph());
}

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

void Op::removeOutDerivedRef(OutIndex index, const TensorId &tensorId) {
  verifyValidOutIndex(index);
  auto &v    = derivedRefs_[index.get()];
  auto found = std::find(v.cbegin(), v.cend(), tensorId);
  if (found == v.cend()) {
    std::ostringstream oss;
    oss << "Cannot remove " << tensorId
        << " as an out derived reference output of output #" << index
        << " as it is not currently one. This for op " << *this;
    throw error(oss.str());
  }
  v.erase(found);
}

void Op::insertOutDerivedRef(OutIndex index, const TensorId &tensorId) {
  verifyValidOutIndex(index);

  if (tensorId == TensorId(id(), index)) {
    std::ostringstream oss;
    oss << "Attempt to insert output derived index at index " << index
        << " of " << *this << " to " << tensorId
        << ", but cannot set an output reference of a tensor to the "
           "tensor itself. ";
    throw error(oss.str());
  }
  derivedRefs_[index.get()].push_back(tensorId);
}
TensorIds Op::refsExcludingSelf(OutIndex o) const {
  TensorId rr = rootRef(o);
  auto ders   = graph().computeOp(rr.opId()).derivedRefs(rr.outIndex());
  for (auto &x : ders) {
    if (x == outTensorId(o)) {
      x = rr;
    }
  }
  return ders;
}

std::vector<std::pair<SubGraphId, CalleeIndex>> Op::indexedCallees() const {

  std::vector<std::pair<SubGraphId, CalleeIndex>> cs;
  auto callees_ = callees();

  cs.reserve(callees_.size());
  for (CalleeIndex ci = 0; ci < callees_.size(); ++ci) {
    cs.push_back({callees_.at(ci.get()), ci});
  }

  return cs;
}

void Op::initializeReplicatedSimOut(SimTensorMap &htm) const {

  // no outputs to initialize.
  if (nOutTensors() == 0) {
    return;
  }

  if (isPartiallyHost()) {
    std::ostringstream oss;
    oss << "Unhandled Op (" << *this << ") in initializeReplicatedSimOut. "
        << "This Op is partially on host, "
        << "with DeviceTypes for inputs:";
    poprithms::util::append(oss, inDeviceTypes());
    oss << " and DeviceTypes for outputs:";
    poprithms::util::append(oss, outDeviceTypes());
    oss << ". "
        << "Tensor initialization should go via initializeSimOut for this "
           "case, as this method "
        << "expects the same replication factor between all "
        << "inputs and outputs. ";
    throw error(oss.str());
  }

  const uint64_t rf = [this, &htm]() -> uint64_t {
    if (nInTensors() == 0) {
      if (deviceTypeByUnanimity() == DeviceType::Host) {
        return 1ull;
      }
      return graph().replicationFactor_u64();
    }
    return htm.getNTensorsByUnanimity(inTensorIds());
  }();

  // this vector is indexed as [rf][outIndex]:
  std::vector<HostTensors> tensors_;
  tensors_.reserve(rf);
  for (uint64_t r = 0; r < rf; ++r) {
    tensors_.push_back(initializeOut(htm.getTensors(inTensorIds(), r)));
  }

  // this vector is indexed as [outIndex][rf]:
  std::vector<HostTensors> nxt(nOutTensors());
  for (uint64_t o = 0; o < nOutTensors(); ++o) {
    for (uint64_t r = 0; r < rf; ++r) {
      nxt[o].push_back(tensors_[r][o]);
    }
  }

  for (uint64_t o = 0; o < nOutTensors(); ++o) {
    htm.setValue({id(), o}, nxt[o]);
  }
}

bool Op::schedulableTypeSpecificEqualTo(
    const poprithms::common::schedulable::Op &other) const {

  // guaranteed that this is valid. Should we just drop this assumption, and
  // have a check?
  const auto &rhs = static_cast<const Op &>(other);

  return
      // Same base properties:
      getComputeState() == rhs.getComputeState() &&

      //  Same derived class:
      typeid(*this) == typeid(rhs) &&
      // Same derived class properties:
      computeTypeSpecificEqualTo(rhs);
}

HostTensors Op::badValOuts() const {

  HostTensors outs;
  outs.reserve(nOutTensors());
  for (uint64_t o = 0; o < nOutTensors(); ++o) {

    // Choosing any non-zero value for initializing tensors with:
    auto type  = outDType(o);
    double val = -99.;
    if (poprithms::ndarray::isUnsignedFixedPoint(type)) {
      val = 99;
    }
    if (type == DType::Boolean) {
      val = true;
    }

    outs.push_back(HostTensor::ones(outDType(o), outShape(o)).mul_(val));
  }
  return outs;
}

HostTensors Op::zeroOuts() const {
  HostTensors outs;
  outs.reserve(nOutTensors());
  for (uint64_t o = 0; o < nOutTensors(); ++o) {
    outs.push_back(HostTensor::zeros(outDType(o), outShape(o)));
  }
  return outs;
}

CodeLocation Op::locationByUnanimity() const {
  return poprithms::common::compute::Graph::codeLocationFromDeviceType(
      deviceTypeByUnanimity());
}
void Op::createVariables(MemoryAliasMapper &mag) const {

  poprithms::memory::alias::TensorIds ts;
  ts.reserve(nOutTensors());
  for (auto s : outShapes()) {
    ts.push_back(mag.graph().allocate(s, MemoryAliasVariable));
  }
  mag.insert(ts, outTensorIds());
}

bool Op::gradientPropagates(OutIndex o) const {
  for (InIndex i = 0; i < nInTensors(); ++i) {
    if (gradientPropagates(o, i)) {
      return true;
    }
  }
  return false;
}

void Op::createAlias(MemoryAliasMapper &b, const TensorId &id) const {
  if (nOutTensors() != 1) {
    throw error(
        "createAlias method should only be used for ops with 1 output");
  }
  auto nxt = b.graph().identity(b.id(id));
  b.insert({nxt}, outTensorIds());
}

[[noreturn]] void Op::unimplemented(const std::string &cntxt) const {
  std::ostringstream oss;
  oss << "For Op " << *this << ", unimplemented method. Context=\"" << cntxt
      << "\".";
  throw error(oss.str());
}

[[noreturn]] void Op::invalid(const std::string &cntxt) const {
  std::ostringstream oss;
  oss << "Invalid method called for Op " << *this << ". Context=\"" << cntxt
      << "\".";
  throw error(oss.str());
}

OptionalTensorIds Op::srcsInCallees(OutIndex o) const {
  OptionalTensorIds ids(nCallees());
  for (uint32_t ci = 0; ci < nCallees(); ++ci) {
    if (isCopiedOut(o, ci)) {
      ids[ci] = srcInCallee(o, ci);
    }
  }
  return ids;
}

} // namespace compute
} // namespace common
} // namespace poprithms
