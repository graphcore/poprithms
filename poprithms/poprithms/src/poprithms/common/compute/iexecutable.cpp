// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "error.hpp"

#include <sstream>

#include <poprithms/common/compute/iexecutable.hpp>

namespace poprithms {
namespace common {
namespace compute {

IExecutable::IExecutable(Graph &&m) : graph_(std::move(m)) {}

void IExecutable::run(SubGraphId sgId) {
  if (!graph().isRunnable(sgId)) {
    std::ostringstream oss;
    oss << "Invalid call, run(SubGraphId = " << sgId << "). "
        << "Only the SubGraphIds " << graph().runnable()
        << " have been marked as runnable. ";
    oss << "Use the Graph::setRunnable method "
        << "to specify which SubGraphs can be run. ";
    throw error(oss.str());
  }
  executableSpecificRun(sgId);
}

using HostTensor = poprithms::compute::host::Tensor;

IExecutable::~IExecutable() = default;

HostTensor IExecutable::getHostValue(const TensorId &id) const {
  graph().verifyIsHost(id);
  return executableSpecificGetHostValue(id);
}

HostTensor IExecutable::getRemoteValue(const TensorId &id, uint64_t r) const {
  graph().verifyIsRemote(id);
  if (r >= graph().replicationFactor_u64()) {
    std::ostringstream oss;
    oss << "Error in IExecutable::getRemoveValue(id=" << id << ", r=" << r
        << "). The replication factor 'r' is too large. "
        << "The graph only has replication factor "
        << graph().replicationFactor_u64()
        << " (r must be strictly less than this). ";
    throw error(oss.str());
  }
  return executableSpecificGetRemoteValue(id, r);
}

// one rank-2 Tensor for each of the replicas.
void IExecutable::setRemoteValue(const TensorId &id, const HostTensors &ts) {
  if (ts.size() != graph().replicationFactor_u64()) {
    std::ostringstream oss;
    oss << "Error in IExecutable::setRemoteValue(id=" << id
        << ", ts=vector containing " << ts.size() << " host Tensors). "
        << "Expected one host tensor for each replica, but replication "
           "factor is "
        << graph().replicationFactor_u64() << '.';
    throw error(oss.str());
  }

  for (uint64_t i = 0; i < ts.size(); ++i) {
    setRemoteValue(id, ts[i], i);
  }
}

void IExecutable::setRemoteValue(const TensorId &id,
                                 const HostTensor &t,
                                 uint64_t r) {

  graph().verifyIsRemote(id);
  if (t.shape() != graph().shape(id)) {
    std::ostringstream oss;
    oss << "Error in IExecutable::setRemoteValue(id = " << id
        << ", t = host tensor of shape " << t.shape() << ", r = " << r
        << "). Shape of t expected to be " << graph().shape(id)
        << ", the shape of the tensor with id " << id << " in the Graph. ";
    throw error(oss.str());
  }

  executableSpecificSetRemoteValue(id, t, r);
}

} // namespace compute
} // namespace common
} // namespace poprithms
