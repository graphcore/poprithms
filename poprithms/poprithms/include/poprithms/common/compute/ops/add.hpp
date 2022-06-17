// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_ADD_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_ADD_HPP

#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/ops/elementwise.hpp>

namespace poprithms {
namespace common {
namespace compute {

class Add final : public BinaryElementwiseOutplace {
public:
  Add(const Op::State &s) : BinaryElementwiseOutplace(s) {}

  /**
   * Sum-reduce the gradient of the output to each of the 2 input shapes.
   * */
  static OptionalTensors
  addBackpropagate(const GradOpIns &, const Shape &in0, const Shape &in1);

private:
  /**
   * The gradient is propagated from the output index to both of the input
   * indices.
   * */
  bool gradientPropagates(OutIndex, InIndex) const final { return true; }

  /**
   * Update the single tensor in #outs to be the sum of the 2 tensors in #ins.
   * */
  void compute(const HostTensors &ins, const HostTensors &outs) const final;

  OptionalTensors bprop(const GradOpIns &) const final;

  /**
   * The add op does not add any new attributes to its base class, so the
   * following methods are of the simplest form.
   * */
  std::string typeString() const final { return "Add"; }
  void binaryElementwiseDerivedVerifyValid() const final{};
  bool computeTypeSpecificEqualTo(const Op &) const final { return true; }
  std::vector<InIndex> autodiffRequiredIns() const final { return {}; }
  std::vector<OutIndex> autodiffRequiredOuts() const final { return {}; }
  UpOp cloneWithState(const State &) const final;
};

/**
 * Add the input at index 1 to the input at index 0, inplace.
 * */
class Add_ final : public BinaryElementwiseInplace_ {

public:
  Add_(const Op::State &s) : BinaryElementwiseInplace_(s) {}

private:
  void binaryElementwiseInplaceDerivedVerifyValid() const final{};
  UpOp cloneWithState(const State &) const final;

  std::string typeString() const final { return "Add_"; }
  bool computeTypeSpecificEqualTo(const Op &) const final { return true; }

  /**
   * This op, even though it is inplace, can propagate the output gradient to
   * the 2 inputs.
   * */
  bool gradientPropagates(OutIndex, InIndex) const final { return true; }
  std::vector<InIndex> autodiffRequiredIns() const final { return {}; }
  std::vector<OutIndex> autodiffRequiredOuts() const final { return {}; }
  OptionalTensors bprop(const GradOpIns &) const final;

  void compute(const HostTensors &ins, const HostTensors &outs) const final {
    outs[0].add_(ins[1]);
  }
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
