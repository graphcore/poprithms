// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_AUTODIFF_CORE_SUMMARY_HPP
#define POPRITHMS_AUTODIFF_CORE_SUMMARY_HPP

#include <ostream>
#include <string>

#include <poprithms/autodiff/ids/ids.hpp>

namespace poprithms {
namespace autodiff {
namespace core {

/**
 * A high-level descriptor of the key tensors created during graph
 * differentiation.
 *
 * These tensors have a 1-to-1 correspondence with tensors in the
 * corresponding Objective used.
 * */
class Summary {

public:
  /**
   * The input gradient tensors for backpropagation. These gradient tensors
   * correspond 1:1 with the 'gradsProvidedFor' tensors in the corresponding
   * Objective
   * */
  const TensorIds &gradsIn() const { return gradsIn_; }

  /**
   * The input checkpoints for backpropagation. These non-gradient tensors
   * correspond 1:1 with the 'checkpoints' tensors in the corresponding
   * Objective. If Autodiff::createCheckpointVariables() is false, then these
   * tensors are exactly the same as 'checkpoints' in the corresponding
   * 'Objective', otherwise they are distinct tensors.
   * */
  const TensorIds &checkpointsIn() const { return checkpointsIn_; }

  /**
   * The gradients of the targets, with a 1:1 correspondence of corresponding
   * Objective's 'targets'.
   * */
  const TensorIds &targetGrads() const { return targetGrads_; }

private:
  void setGradsIn(const TensorIds &);
  void setCheckpointsIn(const TensorIds &);
  void setTargetGrads(const TensorIds &);
  TensorIds gradsIn_;
  TensorIds checkpointsIn_;
  TensorIds targetGrads_;

  friend class Autodiff;
};

std::ostream &operator<<(std::ostream &, const Summary &);

} // namespace core
} // namespace autodiff
} // namespace poprithms

#endif
