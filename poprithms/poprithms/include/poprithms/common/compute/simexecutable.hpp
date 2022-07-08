// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_SIMEXECUTABLE_HPP
#define POPRITHMS_COMMON_COMPUTE_SIMEXECUTABLE_HPP

#include <poprithms/common/compute/iexecutable.hpp>

namespace poprithms {
namespace common {
namespace compute {

using poprithms::common::compute::IExecutable;
using poprithms::common::compute::SimTensorMap;

/**
 * A 'simulator' executable. All tensors, including those which are not
 * DeviceType::Host, are stored only on host, and all code is run on host.
 * */
class SimExecutable : public IExecutable {

public:
  SimExecutable() = delete;

  SimExecutable(Graph &&m);
  SimExecutable(const Graph &m) : SimExecutable(Graph(m)) {}

  SimExecutable(SimExecutable &&);
  SimExecutable(const SimExecutable &);

  virtual ~SimExecutable() override;

  /**
   * The schedule of the sub-graph #sgId.
   * */
  const OpIds &schedule(SubGraphId sgId) const { return schedules.at(sgId); }

private:
  void executableSpecificRun(SubGraphId) final;
  HostTensor executableSpecificGetHostValue(const TensorId &) const final;

  HostTensor executableSpecificGetRemoteValue(const TensorId &,
                                              uint64_t r) const final;

  void executableSpecificSetRemoteValue(const TensorId &,
                                        const HostTensor &,
                                        uint64_t replica) final;

  SimTensorMap &vals() { return *allVals_.uptr; }
  const SimTensorMap &vals() const { return *allVals_.uptr; }

  void verifyNotReplicated(const TensorId &) const;

  // All tensor values, on cpu.
  poprithms::util::CopyByClone<SimTensorMap> allVals_;

  // All of the schedules, one for each sub-graph of the graph.
  std::map<SubGraphId, OpIds> schedules;
};
} // namespace compute
} // namespace common
} // namespace poprithms

#endif
