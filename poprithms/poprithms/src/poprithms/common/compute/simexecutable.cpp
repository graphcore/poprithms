// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "error.hpp"

#include <memory>

#include <poprithms/common/compute/scheduler.hpp>
#include <poprithms/common/compute/simexecutable.hpp>
#include <poprithms/util/copybyclone_impl.hpp>

namespace poprithms {
namespace common {
namespace compute {

using poprithms::common::compute::Graph;

SimExecutable::SimExecutable(SimExecutable &&)         = default;
SimExecutable::SimExecutable(const SimExecutable &rhs) = default;

namespace {
SimTensorMap initHostSimTensors(const Graph &m) {

  SimTensorMap htm;

  // initialize all tensor vectors.
  using namespace poprithms::compute;
  for (int64_t opId = 0; opId < m.nxtOpId().get(); ++opId) {
    if (m.isLive(opId)) {
      htm.push_back(std::vector<HostTensors>(m.nOutTensors(opId)));
    } else {
      htm.push_back({});
    }
  }

  // populate all tensor vectors.
  for (auto opId : Scheduler::vanillaLoweringSchedule(m)) {
    m.computeOp(opId).initializeSimOut(htm);
  }

  return htm;
}
} // namespace

void SimExecutable::executableSpecificRun(const SubGraphId subGraphId) {
  const auto schedule =
      Scheduler::vanillaComputeSchedule(graph(), subGraphId);
  using namespace poprithms::schedule;
  for (auto opId : schedule) {
    graph().computeOp(opId).runSim(vals());
  }
}

void SimExecutable::verifyNotReplicated(const TensorId &id) const {
  if (vals().getValue(id).size() != 1) {
    std::ostringstream oss;
    oss << "Error in SimExecutable::verifyNotReplicated(tid=" << id << "). "
        << "The stored entry for this tensor "
        << "has multiple HostTensors, "
        << "suggesting it's replicated. "
        << "This error might be caused by an attempt "
        << "to set/get a non-host tensor. ";
    throw error(oss.str());
  }
}

HostTensor
SimExecutable::executableSpecificGetHostValue(const TensorId &tId) const {
  verifyNotReplicated(tId);
  return vals().getValue(tId)[0];
}

HostTensor
SimExecutable::executableSpecificGetRemoteValue(const TensorId &tId,
                                                uint64_t r) const {
  auto t0 = vals().getValue(tId).at(r);

  // We copy the values, as stipulated in the method definition.
  return t0.copy();
}

void SimExecutable::executableSpecificSetRemoteValue(
    const TensorId &tId,
    const HostTensor &hostTensor,
    uint64_t replica) {
  vals()[tId][replica].copyFrom_(hostTensor);
}

SimExecutable::~SimExecutable() = default;

SimExecutable::SimExecutable(Graph &&m)
    : IExecutable(std::move(m)),
      allVals_(std::make_unique<SimTensorMap>(initHostSimTensors(graph()))) {}

} // namespace compute
} // namespace common
} // namespace poprithms
