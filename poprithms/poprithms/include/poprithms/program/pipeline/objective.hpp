// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_PIPELINE_OBJECTIVE_HPP
#define POPRITHMS_PROGRAM_PIPELINE_OBJECTIVE_HPP

#include <map>
#include <unordered_set>

#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/traversal.hpp>
#include <poprithms/ndarray/deviceid.hpp>

namespace poprithms {
namespace program {
namespace pipeline {

using PipelineStage  = poprithms::util::TypedInteger<'P', int>;
using PipelineStages = std::vector<PipelineStage>;

using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OpIds;
using poprithms::common::multiout::OutIndex;
using poprithms::common::multiout::OutIndices;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using poprithms::ndarray::DeviceId;
using poprithms::ndarray::DeviceIds;

/**
 * A container class defining how to pipeline a graph.
 * */

class Objective {

public:
  const TensorIds &streamingInputs() const { return streamingInputs_; }
  uint64_t nToAccumulate() const { return nToAccumulate_; }

  /**
   * \param stages Which pipeline stages ops should execute in.
   *
   * \param stageDevices Which devices pipeline stages should execute on.
   *
   * \param nToAccumulate The number of samples to stream through the
   *        pipeline. This should be at least as large as the number of
   *        pipeline stages.
   *
   * \param toAccumulate Which tensors should be accumulated. Typically these
   *        are the 'weight gradient' tensors.
   *
   * \param streamingInputs Which input tensors are streaming inputs.
   *        Typically these are 'data' or 'activation' tensors, as opposed to
   *        'weight' tensors.
   * */
  Objective(const std::map<OpId, PipelineStage> &stages,
            const DeviceIds &stageDevices,
            uint64_t nToAccumulate,
            const TensorIds &toAccumulate,
            const TensorIds &streamingInputs);

  /**
   * \return The pipeline stage of the op #opId.
   * */
  PipelineStage stage(OpId opId) const;

  PipelineStage stage(const TensorId &tId) const { return stage(tId.opId()); }

  DeviceId deviceId(PipelineStage ps) const;

  uint64_t nStages() const { return stageDevices_.size(); }

  /**
   * \return All of the pipeline stages, [0,1,2,...nStages).
   * */
  PipelineStages ascendingStages() const;

  bool mustAccumulate(const TensorId &tId) const {
    return mustAccumulate_.count(tId) != 0;
  }

  /**
   * \return All tensors in the pipeline stage #ps which must be accumulated.
   * */
  TensorIds mustAccumulate(PipelineStage ps) const;

private:
  std::map<OpId, PipelineStage> stages_;

  DeviceIds stageDevices_;

  TensorIds toAccumulate_;

  std::unordered_set<TensorId> mustAccumulate_;

  TensorIds streamingInputs_;

  uint64_t nToAccumulate_;
};
} // namespace pipeline
} // namespace program
} // namespace poprithms

#endif
