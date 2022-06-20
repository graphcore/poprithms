// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_AUTODIFF_AUTOMATIC_GRADOPIN_HPP
#define POPRITHMS_AUTODIFF_AUTOMATIC_GRADOPIN_HPP

#include <ostream>
#include <sstream>

#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/optionaltensorid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/error/error.hpp>

namespace poprithms {
namespace autodiff {
namespace automatic {

using poprithms::common::multiout::InIndex;
using poprithms::common::multiout::OptionalTensorId;
using poprithms::common::multiout::OptionalTensorIds;
using poprithms::common::multiout::OutIndex;

/**
 * The inputs to a gradient op.
 *
 * \tparam OptionalTensor a class with the following subset of the API of
 *                        std::optional<Tensor>:
 *                        (1) bool OptionalTensor::has_value()
 *                        (2) Tensor OptionalTensor::value()
 * */
template <class Tensor, class OptionalTensor> class OpIn {

public:
  using OptionalTensors = std::vector<OptionalTensor>;

  /**
   * \param fwdIns the optional inputs of the forward op. Certain ops may
   *               require these values (such as y=sin(x), dy=cos(x)) while
   *               others may not (such as y = exp(x), dy=y).
   *
   * \param fwdOuts the optional outputs of the forward op.
   *
   * \param gradOuts the optional gradients of the outputs of the forward op.
   * */
  OpIn(const OptionalTensors &fwdIns_,
       const OptionalTensors &fwdOuts_,
       const OptionalTensors &gradOuts_)
      : ins(fwdIns_), outs(fwdOuts_), gradOuts(gradOuts_) {

    // Confirm that the number of outputs and gradients of outputs is the
    // same.
    if (gradOuts.size() != outs.size()) {
      std::ostringstream oss;
      oss << "Number of optional outputs, and optional "
          << "gradients of outputs must be the same. "
          << "But number of optional outputs is " << outs.size()
          << ", and number of optional gradients of outputs is "
          << gradOuts.size() << '.';
      throw poprithms::error::error("autodiff::automatic", oss.str());
    }
  }

  bool hasGradOfOutput(OutIndex o) const {
    return gradOuts.at(o.get()).has_value();
  }

  Tensor gradOfOutput(OutIndex o) const {
    return gradOuts.at(o.get()).value();
  }

  bool hasOutput(OutIndex o) const { return outs.at(o.get()).has_value(); }
  Tensor output(OutIndex o) const { return outs.at(o.get()).value(); }

  bool hasInput(InIndex i) const { return ins.at(i.get()).has_value(); }
  Tensor input(InIndex i) const { return ins.at(i.get()).value(); }

  const OptionalTensors &getIns() const { return ins; }
  const OptionalTensors &getOuts() const { return outs; }
  const OptionalTensors &getGradsOfOuts() const { return gradOuts; }

  /**
   * All optional tensors, ins, outs, and grads.
   * */
  OptionalTensors all() const {
    OptionalTensors all;
    all.reserve(ins.size() + outs.size() + gradOuts.size());
    for (const auto &ots : {ins, outs, gradOuts}) {
      all.insert(all.end(), ots.cbegin(), ots.cend());
    }
    return all;
  }

private:
  OptionalTensors ins;
  OptionalTensors outs;
  OptionalTensors gradOuts;
};

} // namespace automatic
} // namespace autodiff
} // namespace poprithms

#endif
