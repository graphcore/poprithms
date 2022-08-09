// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_WITHAUTODIFF_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_WITHAUTODIFF_HPP

#include <poprithms/autodiff/automatic/gradops.hpp>
#include <poprithms/common/compute/gradopins.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/tensor.hpp>
#include <poprithms/common/multiout/ioindices.hpp>

namespace poprithms {
namespace common {
namespace compute {

using common::multiout::InIndex;
using common::multiout::InIndices;
using common::multiout::OutIndex;
using common::multiout::OutIndices;

class AutodiffHelper {
  const Op &op;

public:
  AutodiffHelper(const Op &op_) : op(op_) {}
  uint64_t nInTensors() const { return op.nInTensors(); }

  Shape inShape(InIndex i) const { return op.inShape(i); }
  Shape outShape(OutIndex o) const { return op.outShape(o); }

  DType inDType(InIndex i) const { return op.inDType(i); }
  DType outDType(OutIndex o) const { return op.outDType(o); }

  static Tensor constantLike(const Tensor &t, double v) {
    return t.constant(v);
  }
};

/**
 * An op class which leverages the automatic differentiation methods of
 * another class, AD, to define the autodiff specific virtual methods.
 *
 * \tparam AD the class which defines the autodiff methods.
 *
 * \tparam BaseWithoutAutodiff the class which this class inherits from and
 *                             adds autodiff functionality to.
 * */
template <class AD, class BaseWithoutAutodiff>
class WithAutodiff : public BaseWithoutAutodiff {
public:
  template <typename... Args>
  WithAutodiff(const Op::State &s, const Args &...args)
      : BaseWithoutAutodiff(s, args...) {}

  InIndices autodiffRequiredIns() const final {
    return AD::autodiffRequiredIns();
  }

  OutIndices autodiffRequiredOuts() const final {
    return AD::autodiffRequiredOuts();
  }

  OptionalTensors bprop(const GradOpIns &g) const final {
    return AD::template backpropagate<Tensor, OptionalTensor, AutodiffHelper>(
        g, {*this});
  }

  bool gradientPropagates(OutIndex o, InIndex i) const final {
    return AD::gradientPropagates(o, i);
  }
};

/**
 * For some ops, attempts at differentiation should result in an error. An
 * example is inplace ops which require an input value to compute a gradient
 * of an input -- the input is not available because it gets modified by the
 * op.
 *
 * Note that this is different to ops which always propagate a zero gradient.
 * An example of such an op is one which sets the value of its output to some
 * value which is independent of the input value.
 *
 * This class is for ops which should error when they are differentiated. An
 * example is inplace sin.
 * */
template <class Base> class NoAutodiff : public Base {

public:
  NoAutodiff(const Op::State &s) : Base(s) {}

  InIndices autodiffRequiredIns() const final {
    throw poprithms::error::error("common::compute", whyNoAutodiff());
  }

  OutIndices autodiffRequiredOuts() const final {
    throw poprithms::error::error("common::compute", whyNoAutodiff());
  }

  OptionalTensors bprop(const GradOpIns &) const final {
    throw poprithms::error::error("common::compute", whyNoAutodiff());
  }

  bool gradientPropagates(OutIndex, InIndex) const final {
    throw poprithms::error::error("common::compute", whyNoAutodiff());
  }

private:
  // What is the reason that the op cannot be backpropagated through?
  virtual std::string whyNoAutodiff() const = 0;
};

/**
 * An op which always propagates a zero value for its input gradient(s). This
 * should not be confused with NoAutodiff.
 * */
template <typename Base>
using ZeroAutodiff =
    WithAutodiff<autodiff::automatic::ZeroPropagationAutodiffer, Base>;

/**
 * Ops without any attributes can use generic clone and comparison methods.
 * Ops inheriting from this class must have the static member OP::OpTypeName
 * defined.
 * */
template <class Base, class OP> class Attributeless : public Base {
public:
  Attributeless(const Op::State &s) : Base(s) {}

private:
  UpOp cloneWithState(const Op::State &s) const final {
    return std::unique_ptr<Op>(new OP(s));
  }

  bool computeTypeSpecificEqualTo(const Op &) const final { return true; }

  std::string typeString() const final { return OP::OpTypeName; }

  void computeDerivedRemoveInputs(const ContiguousInIndexSubset &) final {}
  void computeDerivedRemoveOutputs(const ContiguousOutIndexSubset &) final {}
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
