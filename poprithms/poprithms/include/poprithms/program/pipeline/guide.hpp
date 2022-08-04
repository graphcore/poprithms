// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_PIPELINE_GUIDE_HPP
#define POPRITHMS_PROGRAM_PIPELINE_GUIDE_HPP

#include <unordered_set>

#include <poprithms/program/pipeline/iquerier.hpp>
#include <poprithms/program/pipeline/objective.hpp>

namespace poprithms {
namespace program {
namespace pipeline {

/**
 * This class provides methods to partition ops and tensors into 2 sets. One
 * set contains 'changing' tensors (on a path from a streamed input) and the
 * other contains 'unchanging' tensors.
 *
 * Typically for ML applications, the 'changing' set is 'data' and
 * 'activations', as opposed to model 'weights'.
 *
 * The distinction is useful, as tensors whose value never changed do not need
 * stashes which store multiple values for for future pipeline stages. If an
 * 'unchanging' tensor is used in different pipeline stages on different
 * devices, it can be copied to all locations where it is used just once,
 * before any streamed inputs enter the pipeline.
 * */
class Guide {

public:
  /**
   * \param objective A pipelining objective, which defines which ops belong
   *        in which stages, and which devices the stages should execute on.
   *
   * \param querier Contains basic graph information, such as which tensors
   *        are consumed by which ops. This interface is implemented by the
   *        user of this project.
   * */
  Guide(const IQuerier &querier, const Objective &objective);

  /**
   * \return true if the tensor #tId is NOT on a path from a streamed input.
   *         In ML terms this is typically true of 'model weights', and any
   *         tensors which are derived exclusively from them.
   * */
  bool isUnchanging(const TensorId &tId) const;

  /**
   * \return All of the pipeline stages which have an op which consumes the
   *         tensor #tId. The returned stages are ordered from lowest to
   *         highest. The pipeline stage of #tId is not included, and so if
   *         all consumers of #tId are in the same pipeline state as #tId then
   *         the returned vector is empty.
   * */
  PipelineStages consumers(const TensorId &tId) const;

  /**
   * \return false if any input or output of #opId is changing, and true
   *         otherwise.
   * */
  bool isUnchanging(OpId opId) const;

  /**
   * \return All unchanging ops in the pipeline #ps, in a valid topological
   *        order.
   * */
  OpIds unchangingScheduled(PipelineStage ps) const;

  /**
   * \return All changing ops in the pipeline #ps, in a valid topological
   *        order.
   * */
  OpIds changingScheduled(PipelineStage ps) const;

  /**
   * \return All tensors whose producers are in the pipeline stage #ps.
   * */
  TensorIds tensorIds(PipelineStage ps) const;

  /**
   * \return All ops in the pipeline stage #ps, in a valid topological order.
   * */
  const OpIds &scheduled(PipelineStage ps) const {
    return stageSchedules_.at(ps.get());
  }

private:
  // For each pipeline stage, the ops in topological order.
  std::vector<OpIds> stageSchedules_;

  // All of the tensors which are on a path from a streamed input.
  std::unordered_set<TensorId> fromStreamIn;

  const IQuerier &querier;
  const Objective &objective;
};

} // namespace pipeline
} // namespace program
} // namespace poprithms

#endif
