// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <sstream>

#include <common/compute/error.hpp>

#include <poprithms/autodiff/automatic/call.hpp>
#include <poprithms/autodiff/automatic/repeat.hpp>
#include <poprithms/common/compute/autodiff/automaticmutator.hpp>
#include <poprithms/common/compute/autodiff/automaticquerier.hpp>
#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/ops/withcallees.hpp>
#include <poprithms/common/compute/opverifier.hpp>
#include <poprithms/common/multiout/skiptraversal.hpp>
#include <poprithms/common/multiout/traversal.hpp>

namespace poprithms {
namespace common {
namespace compute {

namespace {

template <typename T> void assertSizes(const T &src, const T &dst) {

  if (src.size() != dst.size()) {
    std::ostringstream oss;
    oss << "Failure in assertSizes, src have " << src.size()
        << " Tensors but dst has " << dst.size() << " Tensors. ";
    throw error(oss.str());
  }
}
} // namespace

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

  OpVerifier(*this).verifyFromAtts({});

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
  withCalleesDerivedRemoveOutputs(coin);
  outs_.reduce(coin);
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
  if (i.get() >= nInputsCopiedToCallees()) {
    std::ostringstream oss;
    oss << "Invalid input index " << i << " in dstInCallee of op " << *this
        << " which only has " << nInputsCopiedToCallees()
        << " input copies. ";
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

  withCalleesDerivedRemoveInputs(coin);
  InIndices copyInIndices = calleeCopyInIndices();
  coin.reduce(inDsts_, copyInIndices);
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

void WithCallees::runSim(ISimState &hts) const {
  computeDerivedVerifyValid();
  hostRun(SimHostRunner(hts));
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
  for (uint64_t i = nInputsCopiedToCallees(); i < nInTensors(); ++i) {
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

  //  All inputs to a call op are copied to the callee sub-graph. In other
  //  words, there is no input like the switch op's #condition argument which
  //  is not copied to the callee sub-graph.
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

bool Repeat::isFlatOut(const TensorId &tId) const {
  if (!outs().isSource(CalleeIndex(0), tId)) {
    return false;
  }
  return isFlatOut(outs().outIndex(CalleeIndex(0), tId));
}

bool Repeat::isStackedOut(const TensorId &tId) const {
  if (!outs().isSource(CalleeIndex(0), tId)) {
    return false;
  }
  return isStackedOut(outs().outIndex(CalleeIndex(0), tId));
}

uint64_t Repeat::getIndexInCarriedTo(const TensorId &tId) const {
  for (uint64_t i = 0; i < carriedTos_.size(); ++i) {
    if (carriedTos_[i] == tId) {
      return i;
    }
  }
  std::ostringstream oss;
  oss << "No tensor " << tId << " is carried to for this op, " << *this;
  throw error(oss.str());
}

uint64_t Repeat::getIndexInCarriedFrom(const TensorId &tId) const {
  for (uint64_t i = 0; i < carriedFroms_.size(); ++i) {
    if (carriedFroms_[i] == tId) {
      return i;
    }
  }
  std::ostringstream oss;
  oss << "No tensor " << tId << " is carried from for this op, " << *this;
  throw error(oss.str());
}

bool Repeat::isCarriedTo(const TensorId &tId) const {
  return std::find(carriedTos_.cbegin(), carriedTos_.cend(), tId) !=
         carriedTos_.cend();
}

bool Repeat::isCarriedFrom(const TensorId &tId) const {
  return std::find(carriedFroms_.cbegin(), carriedFroms_.cend(), tId) !=
         carriedFroms_.cend();
}

bool Repeat::isCarriedIn(InIndex i) const { return !inputIsStackedCopy(i); }

bool Repeat::isFlatOut(OutIndex o) const { return !outputIsStackedCopy(o); }

bool Repeat::isStackedIn(InIndex i) const { return !isCarriedIn(i); }

bool Repeat::hasStackedInIndices() const {
  for (InIndex i = 0; i < nInTensors(); ++i) {
    if (inputIsStackedCopy(i)) {
      return true;
    }
  }
  return false;
}

bool Repeat::hasStackedOutIndices() const {
  for (OutIndex o = 0; o < nOutTensors(); ++o) {
    if (outputIsStackedCopy(o)) {
      return true;
    }
  }
  return false;
}

InIndices Repeat::stackedInIndices() const {
  InIndices inds;
  for (InIndex i = 0; i < nInTensors(); ++i) {
    if (inputIsStackedCopy(i)) {
      inds.push_back(i);
    }
  }
  return inds;
}

InIndices Repeat::carriedInIndices() const {
  InIndices inds;
  for (InIndex i = 0; i < nInTensors(); ++i) {
    if (!inputIsStackedCopy(i)) {
      inds.push_back(i);
    }
  }
  return inds;
}

OutIndices Repeat::flatOutIndices() const {
  OutIndices inds;
  for (OutIndex o = 0; o < nOutTensors(); ++o) {
    if (!outputIsStackedCopy(o)) {
      inds.push_back(o);
    }
  }
  return inds;
}

OutIndices Repeat::stackedOutIndices() const {
  OutIndices inds;
  for (OutIndex o = 0; o < nOutTensors(); ++o) {
    if (outputIsStackedCopy(o)) {
      inds.push_back(o);
    }
  }
  return inds;
}

template <class Accept>
std::set<TensorId> Repeat::visitedBwdFrom(const TensorIds &tIds,
                                          const Accept &a) const {
  using namespace poprithms::common::multiout;
  return depthFirstBwdWithSkips(
      *this, computeGraph(), tIds, a, repeatCount());
}

bool Repeat::gradientPropagates(OutIndex outIndex, InIndex inIndex) const {

  if (computeGraph().isFixedPoint(dstInCallee(inIndex).tId()) ||
      computeGraph().isFixedPoint(
          outs().outSource(outIndex, CalleeIndex(0)))) {
    return false;
  }

  auto dsts = gradientPropagatesFwdFrom({inIndex});
  return dsts.count(outs().outSource(outIndex, CalleeIndex(0))) != 0;
}

std::string Repeat::repeatString() const {
  std::ostringstream oss;
  appendWithCalleesAttributes(oss);
  for (uint64_t n = 0; n < nCarriedTensors(); ++n) {
    oss << "\n       " << carriedTos_[n] << " <--- " << carriedFroms_[n]
        << " (carry back)";
  }
  oss << "\n";

  return oss.str();
}

TensorIds Repeat::carriedFroms(const TensorIds &carriedTos) const {
  TensorIds froms;
  froms.reserve(carriedTos.size());
  for (const auto &tId : carriedTos) {
    froms.push_back(carriedFrom(tId));
  }
  return froms;
}

UpOp Repeat::cloneWithState(const State &s) const {
  return std::make_unique<Repeat>(s,
                                  callee(CalleeIndex(0)),
                                  repeatCount(),
                                  inTensorIdDsts(),
                                  outs().outSources(CalleeIndex(0)),
                                  carriedFroms_,
                                  carriedTos_,
                                  sto_);
}

bool Repeat::withCalleesTypeSpecificEqualTo(const compute::Op &rhs) const {
  const auto &rhs_ = static_cast<const Repeat &>(rhs);
  return repeatCount() == rhs_.repeatCount() &&
         carriedFroms_ == rhs_.carriedFroms_ &&
         carriedTos_ == rhs_.carriedTos_ && sto_ == rhs_.sto_;
}

void Repeat::withCalleesTypeSpecificAssertValid() const {

  //  All inputs to a repeat op are copied to the callee sub-graph. In other
  //  words, there is no input like the switch op's #condition argument which
  //  is not copied to the callee sub-graph.
  if (inDsts().size() != nInTensors()) {
    std::ostringstream oss;
    oss << "This repeat op " << id() << " has " << nInTensors()
        << " input tensors (in caller) but " << inDsts().size()
        << " copy destinations (in callee).";
    throw error(oss.str());
  }

  if (carriedFroms_.size() != carriedTos_.size()) {
    std::ostringstream oss;
    oss << "The repeat op " << id() << " has " << carriedFroms_.size()
        << " carry sources and " << carriedTos_.size()
        << " carry destinations. These should be the same.";
    throw error(oss.str());
  }

  for (uint64_t n = 0; n < carriedFroms_.size(); ++n) {
    if (computeGraph().tensorInfo(carriedFroms_[n]) !=
        computeGraph().tensorInfo(carriedTos_[n])) {
      std::ostringstream oss;
      oss << "The carried tensors #" << n << " (" << carriedFroms_[n]
          << " --> " << carriedTos_[n]
          << ") do not have the same tensor information. "
          << computeGraph().tensorInfo(carriedFroms_[n])
          << " != " << computeGraph().tensorInfo(carriedTos_[n]) << ".";
      throw error(oss.str());
    }

    if (computeGraph().subGraphId(carriedTos_[n]) != callee(CalleeIndex(0))) {
      std::ostringstream oss;
      oss << "The carry tensor destination " << carriedTos_[n]
          << " is not in the callee sub-graph of op " << id();
      throw error(oss.str());
    }
  }

  for (OutIndex o = 0; o < nOutTensors(); ++o) {
    auto &&srcShape_ =
        computeGraph().shape(outs().outSource(o, CalleeIndex(0)));

    if (!isFlatOut(o)) {
      verifyFirstIsSecondStacked(outTensorId(o),
                                 outs().outSource(o, CalleeIndex(0)));
    }

    else {
      if (outShape(o) != srcShape_) {
        std::ostringstream oss;
        oss << "Flat outputs must have the same shape, " << outShape(o)
            << " != " << srcShape_ << " at index " << o << " of " << *this
            << '.';
        throw error(oss.str());
      }
    }
  }

  for (InIndex i = 0; i < nInTensors(); ++i) {
    auto &&dstShape_ = computeGraph().shape(dstInCallee(i).tId());
    if (isStackedIn(i)) {
      verifyFirstIsSecondStacked(inTensorId(i), dstInCallee(i).tId());
    } else {
      if (inShape(i) != dstShape_) {
        std::ostringstream oss;
        oss << "Flat inputs must have the same shape, " << inShape(i)
            << " != " << dstShape_ << " at index " << i << " of " << *this
            << '.';
        throw error(oss.str());
      }
    }
  }
}

std::string Repeat::typeString() const {
  std::ostringstream oss;
  oss << "Repeat(id=" << callee(CalleeIndex(0))
      << ",repeatCount=" << repeatCount();

  if (nCarriedTensors() > 0) {
    std::vector<std::string> carryStrings;
    for (uint64_t i = 0; i < nCarriedTensors(); ++i) {
      oss << ",";
      carryStrings.push_back(carriedTos_[i].str() + "<-" +
                             carriedFroms_[i].str());
    }
    oss << "carries=";
    poprithms::util::append(oss, carryStrings);
  }

  oss << ')';
  return oss.str();
}

bool Repeat::definitelySameValueEveryIteration(const TensorId &tId) const {

  // All the tensors leading to #tId from graph inputs.
  auto searchBack = depthFirstBackwardTensors(
      computeGraph(), {tId}, [](const auto &) { return true; });

  // Are any of the tensors leading to #tId stacked inputs? If so, then the
  // value depends on the slice of the stacked input used.
  for (auto enRoute : searchBack) {
    if (isDstInCallee({enRoute, CalleeIndex(0)})) {
      auto indexIn = inIndex({enRoute, CalleeIndex(0)});
      if (isStackedIn(indexIn)) {
        return false;
      }
    }
  }

  // Are any of the tensors leading to #tId carried-to inputs? If so, and if
  // they're carried to from a tensor which might have a different value (we
  // conservatively check that the carry source is just not the same tensor).
  // then the value of #tId depends on the iteration.
  for (auto enRoute : searchBack) {
    if (isCarriedTo(enRoute) && carriedFrom(enRoute) != enRoute) {
      return false;
    }
  }

  return true;
}

void Repeat::withCalleesDerivedRemoveInputs(
    const ContiguousInIndexSubset &coin) {

  TensorIds updatedCarriedFroms;
  TensorIds updatedCarriedTos;
  for (InIndex i = 0; i < nInTensors(); ++i) {
    if (!coin.isRemoved(i) && isCarriedIn(i)) {
      auto index = getIndexInCarriedTo(dstInCallee(i).tId());
      updatedCarriedTos.push_back(carriedTos_[index]);
      updatedCarriedFroms.push_back(carriedFroms_[index]);
    }
  }

  // replace the old carry tensors with the new ones.
  std::swap(carriedFroms_, updatedCarriedFroms);
  std::swap(carriedTos_, updatedCarriedTos);
}

void Repeat::verifyFirstIsSecondStacked(const TensorId &stacked,
                                        const TensorId &unstacked) const {

  const Shape stackedShape   = computeGraph().shape(stacked);
  const Shape unstackedShape = computeGraph().shape(unstacked);
  poprithms::autodiff::automatic::IRepeatQuerier::verifyFirstIsSecondStacked(
      repeatCount(), stackedShape, unstackedShape);
}

class RepeatHelper final
    : public poprithms::autodiff::automatic::IRepeatQuerier {
  const Repeat &r;

public:
  RepeatHelper(const Repeat &r_) : r(r_) {}

  ~RepeatHelper() override = default;

  StackedCopyOrder stackedCopyOrder() const final {
    return r.stackedCopyOrder();
  }

  // repeat specific:
  OutIndices stackedOutIndices() const final { return r.stackedOutIndices(); }
  OutIndices flatOutIndices() const final { return r.flatOutIndices(); }
  bool definitelySameValueEveryIteration(const TensorId &tId) const final {
    return r.definitelySameValueEveryIteration(tId);
  }
  TensorId carriedTo(const TensorId &tId) const final {
    return r.carriedTo(tId);
  }

  TensorId carriedFrom(const TensorId &tId) const final {
    return r.carriedFrom(tId);
  }
  bool isCarriedFrom(const TensorId &tId) const final {
    return r.isCarriedFrom(tId);
  }

  bool isCarriedTo(const TensorId &tId) const final {
    return r.isCarriedTo(tId);
  }
  bool isStackedIn(InIndex i) const final { return r.isStackedIn(i); }
  bool isStackedOut(const TensorId &tId) const final {
    return r.isStackedOut(tId);
  }
  uint64_t repeatCount() const final { return r.repeatCount(); }
};

poprithms::autodiff::guide::Objective
Repeat::localObjective(CalleeIndex calleeIndex,
                       const InIndices &fromTargets,
                       const OutIndices &inGrads) const {

  if (calleeIndex != 0) {
    throw error("callee index should be 0 in Repeat::localObjective");
  }

  AutomaticQuerier q(static_cast<const Graph &>(computeGraph()));
  const auto r = RepeatHelper(*this);
  return poprithms::autodiff::automatic::RepeatDifferentiator(id(), r, q)
      .createLocalObjective(fromTargets, inGrads);
}

OptionalTensorIds Repeat::growInGrads(
    Graph &machine,
    const poprithms::autodiff::core::ToGradGraph &toGradGraph,
    const poprithms::autodiff::automatic::GradInfos &gradInfos,
    SubGraphId toExtend) const {
  AutomaticMutator gm(machine);
  AutomaticQuerier gq(machine);
  RepeatHelper rp(*this);
  auto outs =
      poprithms::autodiff::automatic::RepeatDifferentiator(id(), rp, gq)
          .createInGrads(gm, toGradGraph, gradInfos, toExtend);
  return outs;
}

CallEvent Repeat::event() const {
  return CallEvent(id(), callee(CalleeIndex(0)), CalleeIndex(0));
}

void Repeat::hostRun(const IHostRunner &fb) const {

  fb.copies(inTensorIds(carriedInIndices()), inDsts(carriedInIndices()));

  for (uint64_t iter = 0; iter < repeatCount(); ++iter) {

    auto stackIndex = stackedCopyOrder() == StackedCopyOrder::Up
                          ? iter
                          : repeatCount() - 1 - iter;

    for (auto i : stackedInIndices()) {
      auto tSource = fb.tensor(inTensorId(i));
      auto tDst    = fb.tensor(dstInCallee(i).tId());
      assertSizes(tSource, tDst);
      const auto rf = tSource.size();
      for (uint64_t ri = 0; ri < rf; ++ri) {
        tDst[ri].update_(tSource[ri].at(stackIndex));
      }
    }

    fb.run(callee(CalleeIndex(0)));

    // We must make sure this happens before the roll copies back.
    for (auto o : stackedOutIndices()) {

      auto tSource = fb.tensor(outs().outSource(o, CalleeIndex(0)));
      auto tDst    = fb.tensor(outTensorId(o));
      assertSizes(tSource, tDst);
      const auto rf = tSource.size();
      for (uint64_t ri = 0; ri < rf; ++ri) {
        tDst[ri].at_(stackIndex).update_(tSource[ri]);
      }
    }

    for (uint64_t n = 0; n < nCarriedTensors(); ++n) {
      fb.copy(carriedFroms_[n], carriedTos_[n]);
    }
  }

  fb.copies(outs().outSources(CalleeIndex(0), flatOutIndices()),
            outTensorIds(flatOutIndices()));
}

template <class Accept>
std::set<TensorId> Repeat::visitedFwdFrom(const TensorIds &tIds,
                                          const Accept &a) const {
  using namespace poprithms::common::multiout;
  return depthFirstFwdWithSkips(
      *this, computeGraph(), tIds, a, repeatCount());
}

std::set<TensorId>
Repeat::gradientPropagatesFwdFrom(const InIndices &inIndices) const {
  auto accept = [this](const auto &x) {
    return computeGraph().gradientPropagates(x);
  };
  return visitedFwdFrom(inDsts(inIndices), accept);
}

TensorIds
Repeat::gradientPropagationVisits(const InIndices &inIndices,
                                  const OutIndices &outIndices) const {
  AutomaticQuerier q(static_cast<const Graph &>(computeGraph()));
  RepeatHelper r(*this);
  return poprithms::autodiff::automatic::RepeatDifferentiator(id(), r, q)
      .gradientPropagationVisits(inIndices, outIndices);
}

} // namespace compute
} // namespace common
} // namespace poprithms
