// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_WITHAUTODIFF_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_WITHAUTODIFF_HPP

#include <poprithms/autodiff/automatic/gradops.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/ops/withoutcallees.hpp>
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
  WithAutodiff(const Op::State &s) : BaseWithoutAutodiff(s) {}

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

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
