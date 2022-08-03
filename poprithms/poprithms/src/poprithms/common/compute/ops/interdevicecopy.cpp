// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <sstream>

#include <common/compute/error.hpp>

#include <poprithms/common/compute/ops/interdevicecopy.hpp>
#include <poprithms/ndarray/dtype.hpp>

namespace poprithms {
namespace common {
namespace compute {

bool CopyBetweenHostAndIpu_::computeTypeSpecificEqualTo(
    const compute::Op &rhs) const {
  const auto &rhs_ = static_cast<const CopyBetweenHostAndIpu_ &>(rhs);
  return copyOptions() == rhs_.copyOptions();
}

void CopyBetweenHostAndIpu_::initializeSimOut(SimTensorMap &htm) const {
  htm.insertCounter(id(), circularBufferCount().get());
  auto value = htm.getValue(destinationId());
  htm.setValue({id(), 0}, value);
}

void CopyBetweenHostAndIpu_::runSim(ISimState &iss) const {
  auto &hts = iss.simTensorMap();
  for (uint64_t r = 0; r < computeGraph().replicationFactor_u64(); ++r) {
    // call into the virtual method which copies either to or from host.
    runCopyHostSim(
        hts[sourceId()], hts[destinationId()], r, hts.getCounterState(id()));
  }

  // increment the index of the circular buffer.
  hts.incrementCounter(id());
}

CopyBetweenHostAndIpu_::CopyBetweenHostAndIpu_(
    const Op::State &s,
    const CopyBetweenHostAndIpuOptions &copyOptions)
    : WithoutCalleesTensorCentric(s), copyOptions_(copyOptions) {}

void CopyBetweenHostAndIpu_::compute(const HostTensors &,
                                     const HostTensors &) const {
  std::ostringstream oss;
  oss << "The op " << *this
      << " is a CopyBetweenHostAndIpu_ op, which means that "
      << "the compute method is not used "
      << "(runSim implemented directly).";
  throw error(oss.str());
}
HostTensors CopyBetweenHostAndIpu_::initializeOut(const HostTensors &) const {
  std::ostringstream oss;
  oss << "The op " << *this
      << " is a CopyBetweenHostAndIpu_ op, which means that "
      << "the initializeOut method is not used "
      << "(initializeSimOut implemented directly).";
  throw error(oss.str());
}

bool CopyBetweenHostAndIpu_::gradientPropagates(OutIndex o, InIndex i) const {
  // The gradient propagates to the source of the copy. See the explanation in
  // automatic::CopyAutodiffer.
  return poprithms::autodiff::automatic::CopyAutodiffer<
      SourceIndex>::gradientPropagates(o, i);
}

OptionalTensors CopyFromIpuToHost_::bprop(const GradOpIns &gIn) const {

  const auto ipuId    = inDeviceId(Source());
  const auto hostGrad = gIn.gradOfOutput(0);

  // Use the same options in the other direction
  auto &&revOptions = copyOptions();

  OptionalTensors gradIns(nInTensors());
  gradIns[Source().get()] = hostGrad.hostToIpu(ipuId, revOptions);

  return gradIns;
}

OptionalTensors CopyFromHostToIpu_::bprop(const GradOpIns &gIn) const {

  OptionalTensors gradIns(nInTensors());
  const auto ipuGrad = gIn.gradOfOutput(0);

  // Use the same options in the other direction
  auto &&revOpts = copyOptions();

  gradIns[Source().get()] = ipuGrad.ipuToHost(circularBufferCount(), revOpts);
  return gradIns;
}

UpOp CopyFromHostToIpu_::cloneWithState(const State &s) const {
  return std::make_unique<CopyFromHostToIpu_>(s, copyOptions());
}

CopyFromHostToIpu_::CopyFromHostToIpu_(
    const State &s,
    const CopyBetweenHostAndIpuOptions &copyOptions)
    : CopyBetweenHostAndIpu_(s, copyOptions) {}

void CopyFromHostToIpu_::runCopyHostSim(
    const HostTensors &src,
    const HostTensors &dst,
    uint64_t replica,
    uint64_t currentCircularBufferIndex) const {

  auto slice = src.at(0).at(currentCircularBufferIndex);
  dst[replica].update_(slice.at(isBroadcast() ? 0 : replica));
}

UpOp CopyFromIpuToHost_::cloneWithState(const State &s) const {
  return std::make_unique<CopyFromIpuToHost_>(s, copyOptions());
}

CopyFromIpuToHost_::CopyFromIpuToHost_(
    const State &s,
    const CopyBetweenHostAndIpuOptions &copyOptions)
    : CopyBetweenHostAndIpu_(s, copyOptions) {}

void CopyFromIpuToHost_::runCopyHostSim(
    const HostTensors &src,
    const HostTensors &dst,
    uint64_t replica,
    uint64_t currentCircularBufferIndex) const {

  dst[0].at_(currentCircularBufferIndex).at_(replica).update_(src[replica]);
}

void CopyBetweenHostAndIpu_::computeDerivedVerifyValid() const {

  if (inDeviceType(hostInputIndex()) != DeviceType::Host) {
    std::ostringstream oss;
    oss << "The device type of input " << hostInputIndex() << " of op "
        << *this << " must be host, but it is "
        << inDeviceType(hostInputIndex());
    throw error(oss.str());
  }

  if (inDeviceType(ipuInputIndex()) != DeviceType::Ipu) {
    std::ostringstream oss;
    oss << "The device type of input " << ipuInputIndex() << " of op "
        << *this << " must be ipu, but it is "
        << inDeviceType(ipuInputIndex());
    throw error(oss.str());
  }

  const auto rf    = computeGraph().replicationFactor_u64();
  auto &&hostShape = inShape(hostInputIndex());
  auto &&ipuShape  = inShape(ipuInputIndex());

  const bool valid = (hostShape.rank_u64() == ipuShape.rank_u64() + 2) &&
                     (hostShape.dim(1) == 1 || hostShape.dim_u64(1) == rf) &&
                     (hostShape.dim(0) > 0) &&
                     (hostShape.fromDim(2) == ipuShape);

  if (!valid) {
    std::vector<std::string> es0{"circularBufferCount(>0)", "1"};
    std::vector<std::string> es1{"circularBufferCount(>0)",
                                 "replicationFactor=" + std::to_string(rf)};

    for (auto x : ipuShape.get()) {
      es0.push_back(std::to_string(x));
      es1.push_back(std::to_string(x));
    }
    std::ostringstream oss;
    oss << "Incompatible tensor shapes for " << *this << ". "
        << "Ipu tensor shape is " << ipuShape << ", and host tensor shape is "
        << hostShape << ". "
        << "With this ipu shape, the expected host shape is ";
    poprithms::util::append(oss, es1);

    if (computeGraph().replicationFactor_u64() > 1) {
      oss << " or ";
      poprithms::util::append(oss, es0);
    }
    oss << ", not " << hostShape << '.' << ". ";
    if (hostShape.rank_u64() != ipuShape.rank_u64() + 2) {
      oss << "The host shape must always have a rank exactly. ";
    }
    throw error(oss.str());
  }
  OpVerifier(*this).verifyNonVariadicFromAtts(
      2, 1, {OpVerifier::Att::SameDType});

  OpVerifier(*this).verifySameTensorInfo(Destination(), 0);
}

/**
 * CopyFromRemoteToIpu_
 * */

void CopyFromRemoteToIpu_::compute(const HostTensors &ins,
                                   const HostTensors &outs) const {
  auto indsVector = ins[Indices().get()].getUnsigned64Vector();
  for (uint64_t i = 0; i < indsVector.size(); ++i) {
    auto src = ins[RemoteSliceable().get()].at(indsVector[i]);
    outs[0].at_(i).update_(src);
  }
}

void CopyFromIpuToRemote_::compute(const HostTensors &ins,
                                   const HostTensors &outs) const {
  auto indsVector = ins[Indices().get()].getUnsigned64Vector();
  for (uint64_t i = 0; i < indsVector.size(); ++i) {
    auto src = ins[IpuSlice().get()].at(i);
    outs[0].at_(indsVector.at(i)).update_(src);
  }
}

HostTensors
CopyBetweenRemoteAndIpu_::initializeOut(const HostTensors &ins) const {
  return {ins[aliasInIndex().get()]};
}

void CopyBetweenRemoteAndIpu_::verifyIndicesShape(const Shape &indices) {
  if (indices.rank_u64() != 1) {
    std::ostringstream oss;
    oss << "The shape of the indices tensor for the remote slice is "
        << indices << " which is rank-" << indices.rank_u64()
        << ". Expected a rank-1 tensor.";
    throw error(oss.str());
  }
}

void CopyBetweenRemoteAndIpu_::computeDerivedVerifyValid() const {

  OpVerifier(*this).verifyNonVariadicFromAtts(3, 1, {});

  // Shapes:

  auto &&remoteShape  = inShape(RemoteSliceable());
  auto &&indicesShape = inShape(Indices());
  auto &&sliceShape   = inShape(IpuSlice());

  verifyIndicesShape(indicesShape);

  if (remoteShape.rank_u64() != 2) {
    std::ostringstream oss;
    oss << "The shape of the remote tensor being sliced is " << remoteShape
        << ". Expected a rank-2 tensor (repeats, numElements) not a rank-"
        << remoteShape.rank_u64() << " tensor.";
    throw error(oss.str());
  }

  {
    auto expected = shapeOfIpuSlice(indicesShape, remoteShape);
    if (sliceShape != expected) {
      std::ostringstream oss;
      oss << "Expected the slice tensor (on ipu) to have shape " << expected
          << " but it has shape " << sliceShape
          << ". This where the indices tensor has shape " << indicesShape
          << " and the remote tensor has shape " << remoteShape << ".";
      throw error(oss.str());
    }
  }

  if (remoteShape.dim(1) != sliceShape.dim(1)) {
    std::ostringstream oss;
    oss << "Remote tensor has shape " << remoteShape
        << " and ipu tensor has shape " << sliceShape
        << ". They should have the same size in dimension 1.";
    throw error(oss.str());
  }

  if (sliceShape.dim(0) != indicesShape.dim(0)) {
    std::ostringstream oss;
    oss << "Slice (ipu) tensor has shape " << sliceShape
        << " and indices tensor has shape " << indicesShape
        << ". They should have the same size in dimension 0.";
    throw error(oss.str());
  }

  // Devices:

  if (inDeviceType(RemoteSliceable()) != DeviceType::Remote) {
    std::ostringstream oss;
    oss << "The device of the input at InIndex=RemoteSliceable is "
        << inDeviceType(RemoteSliceable()) << " not DeviceType::Remote.";
    throw error(oss.str());
  }

  if (inDeviceType(Indices()) != DeviceType::Ipu) {
    std::ostringstream oss;
    oss << "The device of the input at InIndex=Indices is "
        << inDeviceType(Indices()) << " not DeviceType::Ipu.";
    throw error(oss.str());
  }

  if (inDeviceId(IpuSlice()) != inDeviceId(Indices())) {
    std::ostringstream oss;
    oss << "The device of the input at InIndex=Indices is "
        << inDeviceId(IpuSlice())
        << ", not the same as the input at InIndex=IpuSlice, "
        << inDeviceId(IpuSlice()) << ".";
    throw error(oss.str());
  }

  // Types:

  if (inDType(RemoteSliceable()) != inDType(IpuSlice())) {
    std::ostringstream oss;
    oss << "The type of the tensor being sliced (remote tensor) is "
        << inDType(RemoteSliceable()) << " but the destination on ipu is "
        << inDType(IpuSlice()) << ". They must be the same.";
  }

  if (!ndarray::isFixedPoint(inDType(Indices()))) {
    throw error("The Indices input must be fixed point");
  }
}

Shape CopyBetweenRemoteAndIpu_::shapeOfIpuSlice(const Shape &indices,
                                                const Shape &remote) {

  verifyIndicesShape(indices);

  const auto dim0     = 0ULL;
  const auto dim0size = indices.dim(0);
  return remote.resizeSingleDim(dim0size, dim0);
}

Shape CopyBetweenRemoteAndIpu_::shapeOfRemoteSliceable(const Shape &ipuSlice,
                                                       uint64_t nRepeats) {
  if (ipuSlice.rank_u64() != 2) {
    std::ostringstream oss;
    oss << "The ipu (slice) tensor has shape " << ipuSlice
        << ", but to copy to/from remote memory it must be rank-2.";
    throw error(oss.str());
  }

  return {static_cast<int64_t>(nRepeats), ipuSlice.dim(1)};
}

std::string CopyBetweenRemoteAndIpu_::whyNoAutodiff() const {

  std::ostringstream oss;
  oss << "The CopyBetweenHostAndIpu_ op " << *this
      << " does not (currently) support autodiff. "
      << "To support autodiff, we'd need a way of 'summing' remote tensors. "
      << "One way to do this would be to implement the "
      << "core::GraphMutator::sum method for remote tensors to (1) copy "
      << "remote tensors to IPU (2) sum them on IPU (3) copy resulting "
      << "tensor back to remote. Please contact a team member if this is a "
      << "required feature.";
  throw error(oss.str());
}

} // namespace compute
} // namespace common
} // namespace poprithms
