// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <sstream>

#include <common/compute/error.hpp>

#include <poprithms/autodiff/automatic/gradinfos.hpp>
#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/ops/withoutcallees.hpp>

namespace poprithms {
namespace common {
namespace compute {

namespace {
void appendGradOpInIds(std::ostream &ost, const GradOpInIds &gIn) {
  ost << "ins=" << gIn.getIns() << ", outs=" << gIn.getOuts()
      << " gradOuts=" << gIn.getGradsOfOuts();
}
} // namespace

OptionalTensorIds
WithoutCallees::growInGrads(Graph &graph,
                            const ToGradGraph &toGradGraph,
                            const autodiff::automatic::GradInfos &,
                            SubGraphId) const {

  if (&graph != &computeGraph()) {
    throw error("The non-const graph in growInGrads is not the op's graph.");
  }

  const auto optIns         = toGradGraph.optionalNonGrads(inTensorIds());
  const auto optOuts        = toGradGraph.optionalNonGrads(outTensorIds());
  const auto optGradsOfOuts = toGradGraph.optionalGrads(outTensorIds());

  GradOpInIds gradOpIn({optIns, optOuts, optGradsOfOuts});

  auto errorStr = [&gradOpIn, this](const std::string &inOrOut, uint64_t i) {
    std::ostringstream oss;
    oss << "This op, " << *this << ", required " << inOrOut << " #" << i
        << " to compute the gradient of the inputs. "
        << "GradOpInIds ";
    appendGradOpInIds(oss, gradOpIn);
    oss << " does not contain this " << inOrOut << " tensor. ";
    return oss.str();
  };

  // Verify that all inputs required are set:
  for (auto i : autodiffRequiredIns()) {
    if (!gradOpIn.hasInput(i)) {
      throw error(errorStr("input", i.get()));
    }
  }

  // Verify that all outputs required are set:
  for (auto o : autodiffRequiredOuts()) {
    if (!gradOpIn.hasOutput(o)) {
      throw error(errorStr("output", o.get()));
    }
  }

  // Verify that all output gradients required are set:
  for (OutIndex o = 0; o < nOutTensors(); ++o) {

    if (gradientPropagates(o) && !gradOpIn.hasGradOfOutput(o)) {
      std::ostringstream oss;
      oss << "Failure in growInGrads for op" << *this
          << ". Cannot compute gradients of inputs without the "
          << "gradient of output #" << o << '.';
      throw error(oss.str());
    }
  }

  return backpropagate(graph, gradOpIn);
}

void WithoutCallees::invalidAsNoCallees() const {
  std::ostringstream oss;
  oss << "The op " << *this
      << " has no callee sub-graphs, calling this method is an error. ";
  throw error(oss.str());
}

InIndex WithoutCallees::inIndex(const CalleeTensorId &) const {
  invalidAsNoCallees();
}

OutIndex WithoutCallees::outIndex(const CalleeTensorId &) const {
  invalidAsNoCallees();
}

SubGraphId WithoutCallees::callee(CalleeIndex) const { invalidAsNoCallees(); }

CalleeTensorId WithoutCallees::dstInCallee(InIndex) const {
  invalidAsNoCallees();
}

TensorId WithoutCallees::srcInCallee(OutIndex, CalleeIndex) const {
  invalidAsNoCallees();
}

bool WithoutCallees::isDstInCallee(const CalleeTensorId &) const {
  invalidAsNoCallees();
}

bool WithoutCallees::isSrcInCallee(const CalleeTensorId &) const {
  invalidAsNoCallees();
}

TensorIds WithoutCallees::dstsInCallee(const CalleeTensorId &) const {
  invalidAsNoCallees();
}

bool WithoutCallees::isCopiedOut(OutIndex, CalleeIndex) const {
  invalidAsNoCallees();
}

void WithoutCallees::resetCalleeTensorId(InIndex, const CalleeTensorId &) {
  invalidAsNoCallees();
}

void WithoutCallees::resetOutSource(OutIndex, CalleeIndex, const TensorId &) {
  invalidAsNoCallees();
}

void WithoutCallees::extendAutodiffRequiredTensors(
    poprithms::autodiff::automatic::RequiredIds &ts) const {

  for (auto o : autodiffRequiredOuts()) {
    ts.insert({id(), o});
  }
  for (auto i : autodiffRequiredIns()) {
    ts.insert(inTensorId(i));
  }
}

void WithoutCallees::computeWithChecks(const HostTensors &ins,
                                       HostTensors &outs) const {

  if (isInitializingOp()) {
    throw error("initialzing op, does no compute. Error calling this?");
  }

  for (Port p : {Port::In, Port::Out}) {
    const auto &ts = (p == Port::In ? ins : outs);
    if (ts.size() != nTensors(p)) {
      std::ostringstream oss;
      oss << "Invalid number of " << lowercase(p)
          << "puts in Op::computeWithChecks "
          << "for Op " << *this << ". Expected " << nTensors(p)
          << " but received " << ts.size() << '.';
      throw error(oss.str());
    }

    for (uint64_t i = 0; i < nTensors(p); ++i) {
      if (ts[i].shape() != shape(p, i)) {
        std::ostringstream oss;
        oss << "Invalid shape of " << lowercase(p) << "put #" << i
            << " in Op::computeWithChecks, for Op " << *this
            << ". Expected Shape " << shape(p, i) << ", but received "
            << ts[i].shape();
        throw error(oss.str());
      }

      if (ts[i].dtype() != dtype(p, i)) {
        std::ostringstream oss;
        oss << "Invalid dtype of " << lowercase(p) << "put #" << i
            << " in Op::computeWithChecks, for Op " << *this
            << ". Expected dtype " << dtype(p, i) << ", but received "
            << ts[i].dtype();
        throw error(oss.str());
      }
    }
  }

  compute(ins, outs);
}

void WithoutCallees::runReplicatedSim(SimTensorMap &hts) const {
  const auto rf = hts.getNTensorsByUnanimity(inAndOutTensorIds());
  for (uint64_t r = 0; r < rf; ++r) {
    auto inTensors  = hts.getTensors(inTensorIds(), r);
    auto outTensors = hts.getTensors(outTensorIds(), r);
    computeWithChecks(inTensors, outTensors);
  }
}

} // namespace compute
} // namespace common
} // namespace poprithms
