// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_AUTODIFF_AUTOMATIC_AUTOGRADFUNCTION_HPP
#define POPRITHMS_AUTODIFF_AUTOMATIC_AUTOGRADFUNCTION_HPP

#include <memory>
#include <string>
#include <vector>

#include <poprithms/autodiff/automatic/differentiator.hpp>
#include <poprithms/autodiff/automatic/gradinfos.hpp>
#include <poprithms/autodiff/automatic/iautomaticmutator.hpp>
#include <poprithms/autodiff/automatic/iautomaticquerier.hpp>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/program/callstack/carriedtensorid.hpp>

namespace poprithms {
namespace autodiff {
namespace automatic {

using common::multiout::InIndex;
using common::multiout::InIndices;
using common::multiout::OpId;
using common::multiout::OptionalTensorIds;
using common::multiout::OutIndex;
using common::multiout::OutIndices;
using common::multiout::TensorId;
using common::multiout::TensorIds;
using common::schedulable::SubGraphId;
using common::schedulable::SubGraphIds;

/**
 * Inspired by (and based on) PyTorch torch.autograd.Function.
 *
 * */
class AutogradFunction {

public:
  ~AutogradFunction() = default;

  /**
   * \param ad Provides context, and an interface to modify a graph. For
   *        example, it relates tensors to their gradients, and checkpoints to
   *        their sources. This object is required, as unlike with PyTorch
   *        tensors, the TensorIds which this class interacts with contain no
   *        context (a TensorId is just an OpId and an OutIndex, essentially 2
   *        integers).
   * */
  AutogradFunction(Differentiator &ad) : ad_(ad) {}

  Differentiator &differentiator() { return ad_; }

  /**
   * Perform the following steps:
   *
   * (1) Create a sub-graph corresponding to the implementation of the
   *     virtual method #forwards.
   *
   * (2) Create a sub-graph corresponding to the implementation of the
   *     virtual method #backwards.
   *
   * (3) Create a call into (1), without inputs #ins (in the calling
   * sub-graph).
   *
   * (4) register (2) as the gradient sub-graph of (3).
   *
   * \param dbgName A string which is attached to the names of sub-graphs (1)
   *                and (2).
   * */
  TensorIds apply(const TensorIds &ins, const std::string &dbgName);

private:
  /**
   * The forwards computation.
   * */
  virtual TensorIds forwards(const TensorIds &ins) = 0;

  /**
   * The backwards computation.
   *
   * \param fwdOuts The outputs of #forwards.
   *
   * \param outGrads The gradients of the outputs of the forwards computation.
   *        These are optional tensors, because not all outputs are
   *        necessarily on a differentiable path to the loss.
   *
   * \return The gradients of the inputs of #forwards.
   * */
  virtual OptionalTensorIds backwards(const TensorIds &fwdOuts,
                                      const OptionalTensorIds &outGrads) = 0;

  /**
   * The method #backwards uses a subset of the optional gradients of the
   * outputs of #forwards. This method returns true if #backwards uses the
   * output gradient #o.
   *
   * The default behaviour is to return true. This is safe, but requires a
   * pruning pass to remove tensors which are not used in the backwards graph.
   * */
  virtual bool fwdOutGradUsedInBackwards(OutIndex) const { return true; }

  Differentiator &ad_;
};

} // namespace automatic
} // namespace autodiff
} // namespace poprithms

#endif
