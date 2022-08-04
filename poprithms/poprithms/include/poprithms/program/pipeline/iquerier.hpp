// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_PIPELINE_IQUERIER_HPP
#define POPRITHMS_PROGRAM_PIPELINE_IQUERIER_HPP

#include <poprithms/common/multiout/consumptionid.hpp>
#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/traversal.hpp>
#include <poprithms/ndarray/shape.hpp>

namespace poprithms {
namespace program {
namespace pipeline {

using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OpIds;

using poprithms::common::multiout::ConsumptionId;
using poprithms::common::multiout::ConsumptionIds;

using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;

using poprithms::common::multiout::OutIndex;
using poprithms::common::multiout::OutIndices;

using poprithms::ndarray::Shape;
using poprithms::ndarray::Shapes;

/**
 * Interface for querying tensors, required for this projects pipeline
 * transformation.
 * */
class IQuerier {
public:
  virtual ~IQuerier() = default;

  /**
   * The number of outputs of the op #opId.
   * */
  virtual uint64_t nOutTensors(OpId) const = 0;

  /**
   * The ops (and the input indices) which consume the tensor #tId.
   * */
  virtual ConsumptionIds consumptionIds(const TensorId &tId) const = 0;

  /**
   * The ops in the sub-graph being pipelined, in a valid topological order.
   * */
  virtual OpIds schedule() const = 0;

  /**
   * The inputs of the op #opId.
   * */
  virtual TensorIds inTensorIds(OpId) const = 0;

  /**
   * The shape of the tensor #tId.
   * */
  virtual Shape shape(const TensorId &) const = 0;

  /**
   * The output tensor ids. If #opId has n outputs, these are
   * [{opId, 0}, ... {opId, n}). */
  TensorIds outTensorIds(OpId opId) const;

  OutIndices outIndices(OpId) const;
};

} // namespace pipeline
} // namespace program
} // namespace poprithms

#endif
