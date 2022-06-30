// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <sstream>

#include <common/compute/error.hpp>

#include <poprithms/common/compute/ops/encode.hpp>

namespace poprithms {
namespace common {
namespace compute {

HostTensors EncodeOneHot_::initializeOut(const HostTensors &ins) const {
  return {ins[ToEncode().get()]};
}

void EncodeOneHot01_::computeDerivedVerifyValid() const {
  verifyNInAndOutTensors(2, 1);
  OpVerifier(*this).verifyInIsFixedPoint(Indices());
  inShape(ToEncode()).assertOneHotEncodeable(inShape(Indices()));
}

void EncodeOneHotOffOn_::computeDerivedVerifyValid() const {
  verifyNInAndOutTensors(4, 1);
  OpVerifier(*this).verifyInIsFixedPoint(Indices());
  inShape(ToEncode()).assertOneHotEncodeable(inShape(Indices()));
  if (inDType(On()) != inDType(ToEncode()) ||
      inDType(Off()) != inDType(ToEncode())) {
    std::ostringstream oss;
    oss << "\nThe input type of the tensor to encode is "
        << inDType(ToEncode())                                        //
        << "\n The input of of the 'on' value is " << inDType(On())   //
        << "\n The input of of the 'off' value is " << inDType(Off()) //
        << "\n They should all be the same.";
    throw error(oss.str());
  }

  if ((inRank(On()) != 0) || inRank(Off()) != 0) {
    std::ostringstream oss;
    oss << "The rank of the 'on' and 'off' tensors must be 0. "
        << "The shape of the 'on' tensor is " << inShape(On())
        << " and the shape of the 'off tensor is " << inShape(Off());
    throw error(oss.str());
  }
}

void EncodeOneHot01_::compute(const HostTensors &ins,
                              const HostTensors &outs) const {
  outs[0].encodeOneHot_(ins[Indices().get()].getUnsigned64Vector());
}

void EncodeOneHotOffOn_::compute(const HostTensors &ins,
                                 const HostTensors &outs) const {

  // set to have values 0 and 1:
  outs[0].encodeOneHot_(ins[Indices().get()].getUnsigned64Vector());

  // set to have values 0 and On - Off.
  outs[0].mul_(ins[On().get()] - ins[Off().get()]);

  // set to have values Off and On.
  outs[0].add_(ins[Off().get()]);
}
} // namespace compute
} // namespace common
} // namespace poprithms
