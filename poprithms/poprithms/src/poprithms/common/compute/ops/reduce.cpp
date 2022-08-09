// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <sstream>

#include <common/compute/error.hpp>

#include <poprithms/common/compute/ops/reduce.hpp>

namespace poprithms {
namespace common {
namespace compute {

/**
 * Reduce
 * */
HostTensors Reduce::initializeOut(const HostTensors &) const {
  return badValOuts();
}

void Reduce::computeDerivedVerifyValid() const {
  OpVerifier(*this).verifyNonVariadicFromAtts(
      1, 1, {OpVerifier::Att::SameDevice, OpVerifier::Att::SameDType});

  if (inShape(0).rank_u64() != outShape(0).rank_u64()) {
    std::ostringstream oss;
    oss << "The rank of the input is " << inShape(0).rank_u64()
        << ", and the rank of the output is " << outShape(0).rank_u64()
        << ", they should be the same for a reduce op. ";
    throw error(oss.str());
  }
}

bool Reduce::computeTypeSpecificEqualTo(const compute::Op &rhs) const {
  const auto &rhs_ = static_cast<const Reduce &>(rhs);
  return dimensions() == rhs_.dimensions();
}

std::string Reduce::typeString() const {
  std::ostringstream oss;
  oss << "Reduce" << cop() << "(dims=" << dimensions() << ")";
  return oss.str();
}
void Reduce::compute(const HostTensors &ins, const HostTensors &outs) const {
  outs[0].update_(ins[0].reduce(outShape(0), cop()));
}

/**
 * ReduceSum
 * */

UpOp ReduceSum::cloneWithState(const State &s) const {
  return std::make_unique<ReduceSum>(s, dimensions());
}

// TODO(T64547) Inplace expand. Should also consider the gradients generated
// by the view-changing ops themselves.
OptionalTensors ReduceSum::bprop(const GradOpIns &gIn) const {
  auto t0 = gIn.gradOfOutput(0).expand_(inShape(0));
  OptionalTensor t1(t0);
  return {t1};
}

/**
 * ReduceMin
 * */

UpOp ReduceMin::cloneWithState(const State &s) const {
  return std::make_unique<ReduceMin>(s, dimensions());
}

OptionalTensors ReduceMin::bprop(const GradOpIns &gIn) const {
  return {gIn.input(0)
              .equalTo(gIn.output(0))
              .to(outDType(0))
              .mul(gIn.gradOfOutput(0))};
}

/**
 * ReduceMax
 * */

UpOp ReduceMax::cloneWithState(const State &s) const {
  return std::make_unique<ReduceMax>(s, dimensions());
}

OptionalTensors ReduceMax::bprop(const GradOpIns &gIn) const {

  // Note that we don't need an op with 2 outputs. The mask should be
  // generated immediately, and a smart scheduler should push the cast back to
  // the output type as later as possible.
  return {gIn.input(0)
              .equalTo(gIn.output(0))
              .to(outDType(0))
              .mul(gIn.gradOfOutput(0))};
}

/**
 * ReduceProduct
 * */

UpOp ReduceProduct::cloneWithState(const State &s) const {
  return std::make_unique<ReduceProduct>(s, dimensions());
}

OptionalTensors ReduceProduct::bprop(const GradOpIns &) const {
  unimplemented("ReduceProduct::bprop");
}

HostTensors
ReduceAcrossReplicasOutplace::initializeOut(const HostTensors &) const {
  return {HostTensor::zeros(outDType(0), outShape(0))};
}

HostTensors
ReduceAcrossReplicasInplace_::initializeOut(const HostTensors &ins) const {
  return ins;
}

void ReduceAcrossReplicas::compute(const HostTensors &,
                                   const HostTensors &) const {
  throw error("ReduceAcrossReplicas::compute call invalid, as runSim is "
              "implemented directly (access to all replicas required).");
}

void ReduceAcrossReplicas::runSim(ISimState &hts) const {

  auto ins  = hts.simTensorMap().getValue(inTensorId(InIndex(0)));
  auto outs = hts.simTensorMap().getValue(outTensorId(OutIndex(0)));

  auto replicasByGroup = grouping().groups();
  HostTensors reductions;
  reductions.reserve(grouping().nGroups());

  for (auto &&replicas : replicasByGroup) {
    HostTensors ts;
    for (auto r : replicas) {
      ts.push_back(ins.at(r));
    }
    // Accumulate across replicas:
    reductions.push_back(HostTensor::accumulate(ts, cop()));
  }

  // Update local tensors:
  for (uint64_t r = 0; r < outs.size(); ++r) {
    outs.at(r).update_(reductions.at(grouping().group(r)));
  }
}

void ReduceAcrossReplicas::computeDerivedVerifyValid() const {
  OpVerifier(*this).verifyNonVariadicFromAtts(
      1, 1, {OpVerifier::Att::SameDevice, OpVerifier::Att::SameDType});

  OpVerifier(*this).verifyAllIpu();

  if (grouping().range() != replicationFactor_u64()) {
    std::ostringstream oss;
    oss << "Grouping has incorrect replication factor, " << grouping().range()
        << ", expected it to be " << replicationFactor_u64() << '.';
    throw error(oss.str());
  }
}

} // namespace compute
} // namespace common
} // namespace poprithms
