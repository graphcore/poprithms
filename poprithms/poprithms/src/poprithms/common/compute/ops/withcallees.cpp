// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <sstream>

#include <common/compute/error.hpp>

#include <poprithms/autodiff/automatic/call.hpp>
#include <poprithms/common/compute/autodiff/automaticmutator.hpp>
#include <poprithms/common/compute/autodiff/automaticquerier.hpp>
#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/ops/withcallees.hpp>
#include <poprithms/common/compute/opverifier.hpp>
#include <poprithms/common/multiout/traversal.hpp>

namespace poprithms {
namespace common {
namespace compute {

using poprithms::program::callstack::CopyIns;

bool WithCallees::nonRepeatGradientPropagates(const WithCallees &wc,
                                              OutIndex outIndex,
                                              InIndex inIndex) {

  // This is an assumption. An example of this is the #condition input to a
  // switch op, which is not differentiatable. If in the future there's an op
  // with callees and an input which is not copied to a callee which IS
  // differentiable, we'll need to reconsider this. It is hard to imagine such
  // an op.
  if (!wc.isCopyToCalleeInIndex(inIndex)) {
    return false;
  }

  const auto ci = wc.dstInCallee(inIndex).calleeIndex();

  if (!wc.outs().hasValue(outIndex, ci)) {
    return false;
  }

  // See if there's a differentiable path from the output at #outIndex to
  // input at #inIndex.
  const auto inDst  = wc.dstInCallee(inIndex).tId();
  const auto outSrc = wc.outs().outSource(outIndex, ci);

  const auto isDifferentiablePath =
      poprithms::common::multiout::isFwdReachable(
          wc.computeGraph(), {inDst}, outSrc, [&wc](const OpTraversal &x) {
            return wc.computeGraph().gradientPropagates(x);
          });

  return isDifferentiablePath;
}

bool WithCallees::isCallee(SubGraphId sgId) const {
  return std::find(callees_.cbegin(), callees_.cend(), sgId) !=
         callees_.cend();
}

void WithCallees::computeDerivedVerifyValid() const {

  auto base = [this]() {
    std::ostringstream oss;
    appendWithCalleesAttributes(oss);
    return oss.str();
  };

  if (callees_.empty()) {
    throw error(base() + " Callees cannot be empty in WithCallees.");
  }

  // While this doesn't catch all recursion, it catches one of the common
  // mistakes (sub-graph calls itself).
  if (std::find(callees_.cbegin(), callees_.cend(), subGraphId()) !=
      callees_.cend()) {
    std::ostringstream oss;
    oss << base() + " Recursion not supported, but caller (" << subGraphId()
        << ") is in the list of callees (" << callees_ << "). ";
    throw error(oss.str());
  }

  // Check that the sources of the outputs are in the correct sub-graph.
  for (CalleeIndex ci = 0; ci < nCallees(); ++ci) {
    for (OutIndex oi = 0; oi < nOutTensors(); ++oi) {
      if (outs().hasValue(oi, ci)) {
        auto o = outs().outSource(oi, ci);
        if (computeGraph().subGraphId(o) != callee(ci)) {
          std::ostringstream oss;
          oss << base() + " Invalid copy out, " << o
              << " is not in sub-graph " << callee(ci)
              << ", it is in sub-graph " << computeGraph().subGraphId(o);
          throw error(oss.str());
        }
      }
    }
  }

  for (InIndex i = 0; i < inDsts().size(); ++i) {
    if (inDsts_.at(i.get()).calleeIndex() >= callees_.size()) {
      throw error(base() + " Invalid callee index, it's too large.");
    }
  }

  for (InIndex i = 0; i < inDsts().size(); ++i) {
    const auto ci = dstInCallee(i);
    if (computeGraph().subGraphId(ci.tId()) !=
        callees_.at(ci.calleeIndex().get())) {
      throw error(base() +
                  " Disagreement about sub-graph of copy in destination.");
    }
  }

  for (InIndex i = 0; i < inDsts().size(); ++i) {
    const auto ci = dstInCallee(i);
    if (computeGraph().dtype(ci.tId()) !=
        computeGraph().dtype(inTensorId(i))) {
      std::ostringstream oss;
      oss << base() << " Type disagreement at input #i.";
      throw error(oss.str());
    }
  }

  if (outs().nOutTensors() != nOutTensors()) {
    throw error(base() + " Disagreement about number of outputs (currently "
                         "assuming all outputs are copies).");
  }

  if (outs().nOutTensors() > 0) {
    if (outs().nCallees() != callees_.size()) {
      std::ostringstream oss;
      oss << base() << " Number of callees according to outs is "
          << outs().nCallees() << ", but number of callees in callees_ is "
          << callees_.size();
      throw error(oss.str());
    }

    for (OutIndex o = 0; o < nOutTensors(); ++o) {
      for (CalleeIndex c = 0; c < nCallees(); ++c) {
        if (outs().hasValue(o, c)) {
          auto tId = outs().outSource(o, c);
          if (computeGraph().subGraphId(tId) != callee(c)) {
            throw error(base() +
                        " Disagreement about sub-graph of copy out source.");
          }

          if (computeGraph().dtype(tId) != outDType(o)) {
            std::ostringstream oss;
            oss << base() << " Disagreement between source and "
                << "destination of out copy tensor info.";
            oss << "source has info " << computeGraph().tensorInfo(tId)
                << " and destination has info " << outTensorInfo(o) << '.';
            throw error(oss.str());
          }
        }
      }
    }
  }

  OpVerifier(*this).verifyFromAtts({OpVerifier::Att::SameDeviceType});

  for (CalleeIndex ci = 0; ci < nCallees(); ++ci) {
    std::set<TensorId> outsAtIndex;
    for (OutIndex oi = 0; oi < nOutTensors(); ++oi) {
      if (outs().hasValue(oi, ci)) {
        if (outsAtIndex.count(outs().outSource(oi, ci)) != 0) {
          std::ostringstream oss;
          oss << "Output sources must all be distinct at callee index " << ci
              << ". Repeated outputs should be managed with copies/aliases. "
              << "Output sources of " << *this << " for callee index " << ci
              << " are " << outs().outSources(ci);
          throw error(oss.str());
        }
        outsAtIndex.insert(outs().outSource(oi, ci));
      }
    }
  }

  for (InIndex i = 0; i < inDsts().size(); ++i) {
    if (computeGraph().subGraphId(inTensorId(i).opId()) != subGraphId()) {
      std::ostringstream oss;
      oss << base() + " Invalid copy in at index " << i
          << ", src sub-graph not valid for op " << *this;
      throw error(oss.str());
    }
    if (!isCallee(computeGraph().subGraphId(dstInCallee(i).tId().opId()))) {
      std::ostringstream oss;
      oss << base() + " Invalid copy in at index " << i
          << ", dst sub-graph not a callee.";
      throw error(oss.str());
    }
  }

  withCalleesTypeSpecificAssertValid();
}

void WithCallees::resetOutSource(OutIndex o,
                                 CalleeIndex ci,
                                 const TensorId &tId) {

  for (OutIndex o_ = 0; o_ < nOutTensors(); ++o_) {
    if (o != o_ && outs_.hasValue(o_, ci) && outs_.outSource(o_, ci) == tId) {
      std::ostringstream oss;
      oss << "For CalleeIndex " << ci
          << " all the output tensors must be distinct "
          << "at the output indices. "
          << " But if " << tId << " is the new tensor at output index " << o
          << " of " << *this
          << " then it will be the output at at least 2 indices.";
      throw error(oss.str());
    }
  }
  outs_.reset(o, ci, tId);
}

InIndex WithCallees::inIndex(const CalleeTensorId &ctId) const {
  for (InIndex i = 0; i < inDsts_.size(); ++i) {
    if (inDsts_.at(i.get()) == ctId) {
      return i;
    }
  }
  throw error("callee tensor not present");
}

void WithCallees::resetCalleeTensorId(InIndex i, const CalleeTensorId &n) {
  inDsts_.at(i.get()) = n;
}

CodeLocation WithCallees::codeLocation() const {

  // Ensure that inputs and outputs all have same location.
  auto deviceTypes = inDeviceTypes();
  const auto y     = outDeviceTypes();
  deviceTypes.insert(deviceTypes.end(), y.cbegin(), y.cend());

  if (std::set<DeviceType>(deviceTypes.begin(), deviceTypes.end()).size() >
      1) {
    std::ostringstream oss;
    oss << "WithCallees op " << *this
        << ": all inputs and outputs must have same device type. "
        << "DeviceTypes of inputs and outputs are ";
    poprithms::util::append(oss, deviceTypes);
  }

  for (auto c : callees()) {
    for (auto o : computeGraph().opIds(c)) {
      if (computeGraph().computeOp(o).codeLocation() == CodeLocation::Host) {
        return CodeLocation::Host;
      }
    }
  }

  return CodeLocation::Ipu;
}

void WithCallees::computeDerivedRemoveOutputs(
    const ContiguousOutIndexSubset &coin) {
  outs_.reduce(coin);
  withCalleesDerivedRemoveOutputs(coin);
}

std::vector<CopyIn> WithCallees::copyIns() const {
  auto ins_ = inTensorIds();
  ins_      = {ins_.begin(), ins_.begin() + nInCopies()};
  return CopyIns::zip(ins_, inDsts_);
}

bool WithCallees::computeTypeSpecificEqualTo(const Op &rhs) const {
  const auto &rhs_ = static_cast<const WithCallees &>(rhs);
  return callees_ == rhs_.callees() && subGraphId() == rhs_.subGraphId() &&
         inDsts_ == rhs_.inDsts() && outs_ == rhs_.outs() &&
         withCalleesTypeSpecificEqualTo(rhs);
}

HostTensors WithCallees::initializeOut(const HostTensors &) const {
  return badValOuts();
}

CalleeTensorId WithCallees::dstInCallee(InIndex i) const {
  if (i.get() >= nCopyIns()) {
    std::ostringstream oss;
    oss << "Invalid input index " << i << " in dstInCallee of op " << *this
        << " which only has " << nCopyIns() << " input copies. ";
    throw error(oss.str());
  }
  return inDsts_.at(i.get());
}

InIndices WithCallees::calleeCopyInIndices() const {
  auto nons = nonCopyToCalleeIndices();
  if (nons.empty()) {
    return inIndices();
  }
  InIndices inIndices;
  inIndices.reserve(nInTensors() - nons.size());
  for (InIndex i = 0; i < nInTensors(); ++i) {
    if (std::find(nons.cbegin(), nons.cend(), i) == nons.cend()) {
      inIndices.push_back(i);
    }
  }
  return inIndices;
}

void WithCallees::computeDerivedRemoveInputs(
    const ContiguousInIndexSubset &coin) {

  InIndices copyInIndices = calleeCopyInIndices();
  coin.reduce(inDsts_, copyInIndices);
  withCalleesDerivedRemoveInputs(coin);
}

void WithCallees::appendWithCalleesAttributes(std::ostream &ost) const {
  ost << "WithCallees: "
      << "\n      callee=" << callees() << "\n         out=" << outs();
}

bool WithCallees::isSrcInCallee(const CalleeTensorId &tId) const {
  return outs_.isSource(tId.calleeIndex(), tId.tId());
}

TensorId WithCallees::srcInCallee(OutIndex o, CalleeIndex ci) const {
  return outs_.outSource(o, ci);
}

bool WithCallees::isCopiedOut(OutIndex o, CalleeIndex ci) const {
  return outs_.hasValue(o, ci);
}

void WithCallees::extendAutodiffRequiredTensors(
    poprithms::autodiff::automatic::RequiredIds &ts) const {

  // Some tensors in the callee graphs are copied out for use later in the
  // gradient graph. These are the checkpoint tensors.
  for (CalleeIndex calleeIndex = 0; calleeIndex < nCallees(); ++calleeIndex) {
    const auto &gInf = ts.gradInfo(ts.grad(id(), calleeIndex));

    // checkPointPair is a pair of tensors: 1 tensor in the callee graph and 1
    // in the gradient graph. The one in the callee graph is copied out to a
    // tensor #dst in the calling graph.
    for (auto checkPointPair : gInf.checkpointPairs()) {
      if (!outs().isSource(calleeIndex, checkPointPair.inNonGradGraph)) {
        std::ostringstream oss;
        oss << "Expected the tensor " << checkPointPair.inNonGradGraph
            << " to be copied out of op " << *this
            << ", as it is required as a checkpoint. "
            << "In particular, the objective is " << gInf.objective() << '.';
        throw error(oss.str());
      }

      const auto dst = computeGraph().dstInCaller(
          checkPointPair.inNonGradGraph, event(calleeIndex));
      ts.insert(dst);
    }
  }
}

void WithCallees::runSim(SimTensorMap &hts) const {
  computeDerivedVerifyValid();
  hostRun(SimHostRunner(hts, computeGraph()));
}

TensorIds WithCallees::inTensorIdDsts() const {
  TensorIds ids_;
  ids_.reserve(inDsts_.size());
  for (const auto &inDst : inDsts_) {
    ids_.push_back(inDst.tId());
  }
  return ids_;
}

OutIndex WithCallees::outIndex(const CalleeTensorId &x) const {
  return outs_.outIndex(x.calleeIndex(), x.tId());
}

bool WithCallees::isDstInCallee(const CalleeTensorId &cId) const {
  return std::find(inDsts_.cbegin(), inDsts_.cend(), cId) != inDsts_.cend();
}
TensorIds WithCallees::inDsts(const InIndices &inIndices) const {
  TensorIds ids_;
  ids_.reserve(inIndices.size());
  for (auto i : inIndices) {
    ids_.push_back(inDsts_.at(i.get()).tId());
  }
  return ids_;
}

TensorIds WithCallees::inDsts(CalleeIndex ci) const {
  TensorIds ids_;
  for (const auto &ctId : inDsts_) {
    if (ctId.calleeIndex() == ci) {
      ids_.push_back(ctId.tId());
    }
  }
  return ids_;
}

TensorIds WithCallees::inSrcs(CalleeIndex ci) const {

  TensorIds ids_;
  for (InIndex i = 0; i < inDsts_.size(); ++i) {
    if (inDsts_.at(i.get()).calleeIndex() == ci) {
      ids_.push_back(inTensorId(i));
    }
  }
  return ids_;
}

TensorIds WithCallees::dstsInCallee(const CalleeTensorId &inCaller) const {
  TensorIds ids_;
  for (InIndex i = 0; i < inDsts_.size(); ++i) {
    if (inTensorId(i) == inCaller.tId()) {
      auto x = inDsts_.at(i.get());
      if (x.calleeIndex() == inCaller.calleeIndex()) {
        ids_.push_back(x.tId());
      }
    }
  }

  return ids_;
}

WithCallees::WithCallees(const Op::State &opState,
                         const SubGraphIds &callees,
                         const CalleeTensorIds &inDsts,
                         const CopyOuts &outs)
    : Op(opState), callees_(callees), inDsts_(inDsts), outs_(outs) {}

InIndices WithCallees::nonCopyToCalleeIndices() const {
  InIndices inds;
  inds.reserve(nNonCopyIns());
  for (uint64_t i = nCopyIns(); i < nInTensors(); ++i) {
    inds.push_back(InIndex(i));
  }
  return inds;
}

void IHostRunner::copies(const TensorIds &froms, const TensorIds &tos) const {

  if (froms.size() != tos.size()) {
    std::ostringstream oss;
    oss << "froms is of size " << froms.size() << " and tos is of size "
        << tos.size();
    throw error(oss.str());
  }
  for (uint64_t inCopy = 0; inCopy < froms.size(); ++inCopy) {
    copy(froms[inCopy], tos[inCopy]);
  }
}

std::vector<HostTensors> IHostRunner::tensors(const TensorIds &tIds) const {
  std::vector<HostTensors> ts;
  ts.reserve(tIds.size());
  for (auto tId : tIds) {
    ts.push_back(tensor(tId));
  }
  return ts;
}

void IHostRunner::copy(const TensorId &from, const TensorId &to) const {

  const auto tFrom = tensor(from);
  const auto tTo   = tensor(to);
  const auto rf    = tFrom.size();
  if (rf != tTo.size()) {
    throw error("Number of replicas for source and destination differ");
  }
  for (uint64_t replIndex = 0; replIndex < rf; ++replIndex) {
    tTo[replIndex].update_(tFrom[replIndex]);
  }
}

std::string Call::typeString() const {
  return poprithms::util::cat::strcat(
      "Call(callee=", callee(CalleeIndex(0)), ')');
}

Call::Call(const State &s,
           const TensorIds &copyInDsts,
           SubGraphId callee,
           const TensorIds &copyOutSrcs)
    : WithCallees(
          s,
          {callee},
          CalleeTensorId::zip(copyInDsts, CalleeIndex(0)),
          CopyOuts(std::map<CalleeIndex, TensorIds>{{0, copyOutSrcs}})) {}

poprithms::autodiff::guide::Objective
Call::localObjective(CalleeIndex calleeIndex,
                     const InIndices &fromTargets,
                     const OutIndices &inGrads) const {

  if (calleeIndex != 0) {
    throw error("For Call, CalleeIndex should always be 0");
  }

  const auto targets    = inDsts(fromTargets);
  auto gradsProvidedFor = outs().outSources(CalleeIndex(0), inGrads);
  auto checkpoints      = outs().outSources(CalleeIndex(0));

  return poprithms::autodiff::guide::Objective::outOfGraph(
      gradsProvidedFor, checkpoints, targets);
}

UpOp Call::cloneWithState(const State &s) const {
  return std::make_unique<Call>(s,

                                inTensorIdDsts(),
                                callee(CalleeIndex(0)),
                                outs().outSources(CalleeIndex(0)));
}

void Call::withCalleesTypeSpecificAssertValid() const {

  if (inDsts().size() != nInTensors()) {
    std::ostringstream oss;
    oss << "This call op " << id() << " has " << nInTensors()
        << " input tensors (in caller) but " << inDsts().size()
        << " copy destinations (in callee).";
    throw error(oss.str());
  }

  for (InIndex i = 0; i < inDsts().size(); ++i) {
    auto s0 = computeGraph().shape(inTensorId(i));
    auto s1 = computeGraph().shape(dstInCallee(i).tId());
    if (s0 != s1) {
      std::ostringstream oss;
      oss << "Copy in shape mismatch at index " << i << " : " << s0
          << " != " << s1;
      throw error(oss.str());
    }
  }

  for (OutIndex o = 0; o < nOutTensors(); ++o) {
    auto s0 = computeGraph().shape(outTensorId(o));
    auto s1 = computeGraph().shape(outs().outSource(o, CalleeIndex(0)));

    if (s0 != s1) {
      std::ostringstream oss;
      oss << "Copy out shape mismatch at index " << o << " : " << s0
          << " != " << s1;
      throw error(oss.str());
    }
  }
}

bool Call::gradientPropagates(OutIndex outIndex, InIndex inIndex) const {
  return nonRepeatGradientPropagates(*this, outIndex, inIndex);
}

OptionalTensorIds
Call::growInGrads(Graph &graph,
                  const poprithms::autodiff::core::ToGradGraph &toGradGraph,
                  const poprithms::autodiff::automatic::GradInfos &gradInfos,
                  SubGraphId toExtend) const {

  auto gq = AutomaticQuerier(graph);
  auto gm = AutomaticMutator(graph);

  return poprithms::autodiff::automatic::CallDifferentiator::createInGrads(
      id(), gm, gq, toGradGraph, gradInfos, toExtend);
}

void Call::hostRun(const IHostRunner &fb) const {
  fb.copies(inTensorIds(), inTensorIdDsts());
  fb.run(callee(CalleeIndex(0)));
  fb.copies(outs().outSources(CalleeIndex(0)), outTensorIds());
}

} // namespace compute
} // namespace common
} // namespace poprithms
