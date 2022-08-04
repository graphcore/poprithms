// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_PIPELINE_PIPELINE_HPP
#define POPRITHMS_PROGRAM_PIPELINE_PIPELINE_HPP

#include <poprithms/common/multiout/consumptionid.hpp>
#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/traversal.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/program/pipeline/guide.hpp>
#include <poprithms/program/pipeline/imutator.hpp>
#include <poprithms/program/pipeline/iquerier.hpp>
#include <poprithms/program/pipeline/objective.hpp>

namespace poprithms {
namespace program {
namespace pipeline {

class Pipeline {
public:
  /**
   * Create pipeline sub-graphs according to the objective #objective.
   *
   * Note that the graph which is being pipelined is not modified inplace.
   * Rather, a new set of graphs for pipeline stages, inter-device copying,
   * etc. are created.
   *
   * The graphs are created using #mutator, which defines how new ops and
   * sub-graphs are created. The underlying graphs are queried for basic
   * information using #querier.
   * */
  Pipeline(const Objective &objective,
           const IQuerier &querier,
           const IMutator &mutator);

  /**
   * The co-ordinating top-level sub-graph, which calls into the the other
   * sub-graphs.
   * */
  SubGraphId mainPipeline() const { return sgMain; }

  /**
   * The tensor #unpipelined, which was required for accumulation according to
   * #objective, has an accumulation tensor in its pipeline stage. This method
   * return the accumulation tensor.
   * */
  TensorId accumulatorInStage(const TensorId &unpipelined) const;

  /**
   * \return The tensor in the pipeline stage of #unpipelined which
   *         corresponds to it.
   * */
  TensorId getInStage(const TensorId &unpipelined) const {
    return getInStage(objective.stage(unpipelined), unpipelined);
  }

  /**
   * \return The pipeline sub-graph for the stage of op #opId.
   * */
  SubGraphId stageSubGraph(OpId id) const {
    return stageSubGraphs.at(objective.stage(id).get());
  }

private:
  TensorId getInStage(PipelineStage ps, const TensorId &unpipelined) const;

  TensorIds getInStage(PipelineStage ps, const TensorIds &tIds) const;

  TensorIds getUnchangingTensors(const TensorIds &unpipelined,
                                 DeviceId dId) const;

  void setUnchangingTensors(const IMutator &, const IQuerier &);
  void setChangingTensors(const IMutator &, const IQuerier &);
  void setAccumulators(const IMutator &, const IQuerier &);

  Objective objective;
  Guide guide;

  /**
   * The sub-graphs for each pipeline stage.
   * */
  SubGraphIds stageSubGraphs;

  /**
   * For all pipeline stages in [1, nStages), there might be inter-device
   * copies of tensors from previous stages. This sub-graph contains all such
   * copies.
   * */
  SubGraphId sgCopy;

  /**
   * On each device, a scalar integer tensor which keeps track global
   * pipeline iteration.
   * */
  std::map<DeviceId, TensorId> cycleIndices;

  TensorId indexInStage(const IMutator &mutator, PipelineStage ps) {
    return mutator.refFrom_(cycleIndices.at(objective.deviceId(ps)),
                            stageSubGraphs.at(ps.get()));
  }

  /**
   * The accumulator tensors. The keys of this map are tensors in the
   * unpipelined graph, and the values are the accumulator tensors in the
   * pipeline stage sub-graphs.
   * */
  std::map<TensorId, TensorId> accumulators;

  /**
   * The 'main' sub-graph orchestrates the pipeline model. It contains calls
   * for the ramp-up and ramp-down stages and a repeat operation for when the
   * pipeline is full.
   * */
  SubGraphId sgMain;

  /**
   * Does:
   * - restore tensors required in next execution of stages.
   * - stash tensors which have just been computed, for future use.
   * - increment cycle indices.
   * */
  SubGraphId sgRestoreStashTick;

  /**
   * A graph for initializing tensors. It is called once, at the start of the
   * main sub-graph. It
   *  - Sets accumulators (to zero if the accumulation is by summation).
   *  - Sets the cycle indices which tracks pipeline iteration.
   *   - Runs unchanging ops.
   **/
  SubGraphId sgInitialize;

  struct InterStageTensorMapping {
  public:
    // The pipeline stage that a tensor is copied (or referenced) to.
    PipelineStage to;

    // A tensor in a pipeline stage preceding #to. This tensor is restored
    // from a stash in the source pipeline stage, ready to be copied
    // (or referenced) to #destination.
    TensorId restoredSource;

    // A tensor is in the pipeline stage #to.
    TensorId destination;
  };
  using InterStageTensorMappings = std::vector<InterStageTensorMapping>;

  /**
   * The keys of this map are tensors in the unpipelined graph which are
   * consumed on stages other its own. The values, are all the copies
   * (destination pipeline stage and destination tensor) to different pipeline
   * stages.
   * */
  std::map<TensorId, InterStageTensorMappings> interStageTensorMappings;

  /**
   * Unchanging tensors. The keys here are the tensors in the unpipelined
   * graph. The values are the tensors in the initialization sub-graph
   * corresponding to the unpipelined tensor. For tensors which are not
   * consumed in pipeline stages which are on a different device to the
   * producer of the op, there will be just one element in the map value.
   * */
  std::map<TensorId, std::vector<std::pair<DeviceId, TensorId>>>
      unchangingTensorsInInit;

  /**
   * Unchanging tensor references in the pipeline stages.
   * */
  std::map<TensorId, std::vector<std::pair<PipelineStage, TensorId>>>
      unchangingTensorsInPipelineStages;

  /**
   * A map from ops in the unpipelined graph to their equivalents in the
   * sub-graphs for each pipeline stage.
   */
  std::map<OpId, OpId> stageClones;
};

} // namespace pipeline
} // namespace program
} // namespace poprithms

#endif
