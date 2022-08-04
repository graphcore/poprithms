// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_PIPELINE_HPP
#define POPRITHMS_COMMON_COMPUTE_PIPELINE_HPP

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/slickgraph.hpp>
#include <poprithms/common/multiout/consumptionid.hpp>
#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/traversal.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/program/pipeline/guide.hpp>
#include <poprithms/program/pipeline/imutator.hpp>
#include <poprithms/program/pipeline/iquerier.hpp>
#include <poprithms/program/pipeline/objective.hpp>
#include <poprithms/program/pipeline/pipeline.hpp>

namespace poprithms {
namespace common {
namespace compute {

using poprithms::program::pipeline::Objective;
using poprithms::program::pipeline::PipelineStage;
using poprithms::program::pipeline::PipelineStages;

/**
 * Pipeline accumulation type.
 * */
enum class PipelineAcclType { Sum = 0, RunningMean, Max };
using PipelineAcclTypes = std::vector<PipelineAcclType>;

/**
 * An extension to the base Objective class which allows different tensors to
 * be accumulated with different accumulation types.
 * */
class AcclTypedObjective : public Objective {

public:
  AcclTypedObjective(const std::map<OpId, PipelineStage> &stages,
                     const DeviceIds &stageDevices,
                     uint64_t nToAccumulate,
                     const TensorIds &toAccumulate,
                     const PipelineAcclTypes &acclTypes,
                     const TensorIds &streamingInputs);

  AcclTypedObjective(const std::map<OpId, PipelineStage> &stages,
                     const DeviceIds &stageDevices,
                     uint64_t nToAccumulate,
                     const TensorIds &toAccumulate,
                     const TensorIds &streamingInputs)
      : AcclTypedObjective(stages,
                           stageDevices,
                           nToAccumulate,
                           toAccumulate,
                           {toAccumulate.size(), PipelineAcclType::Sum},
                           streamingInputs) {}

  PipelineAcclType acclType(const TensorId &) const;

private:
  std::map<TensorId, PipelineAcclType> acclTypes_;
};

/**
 * Create sub-graphs for a pipelined model of #sgId in #g.
 *
 * Note that currently #sgId cannot contain ops with callees TODO(T66683)).
 * */
class Pipeline : public poprithms::program::pipeline::Pipeline {
public:
  Pipeline(SlickGraph &g, SubGraphId sgId, const AcclTypedObjective &obj);
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
