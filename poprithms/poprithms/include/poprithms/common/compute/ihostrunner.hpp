// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OPS_IHOSTRUNNER_HPP
#define POPRITHMS_COMMON_COMPUTE_OPS_IHOSTRUNNER_HPP

#include <poprithms/common/compute/gradopins.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/scheduler.hpp>
#include <poprithms/common/compute/tensor.hpp>
#include <poprithms/program/callstack/copyin.hpp>
#include <poprithms/program/callstack/copyout.hpp>

namespace poprithms {
namespace common {
namespace compute {

class IHostRunner {
public:
  /**
   * Return the value of the tensor #tId. A vector is returned, not just
   * just a single HostTensor, which allows for replication for the simulator
   * backend.
   * */
  virtual HostTensors tensor(const TensorId &tId) const = 0;
  std::vector<HostTensors> tensors(const TensorIds &) const;

  /**
   * Run the sub-graph #sgId.
   * */
  virtual void run(SubGraphId sgId) const = 0;

  /**
   * Copy the values of the host tensors of #from to those of #to.
   * */
  void copy(const TensorId &from, const TensorId &to) const;
  void copies(const TensorIds &froms, const TensorIds &tos) const;
};

class SimHostRunner final : public IHostRunner {
private:
  SimTensorMap &hts;
  const Graph &m;

public:
  SimHostRunner(SimTensorMap &hts_, const Graph &graph_)
      : hts(hts_), m(graph_) {}

  HostTensors tensor(const TensorId &tId) const final {
    return hts.getValue(tId);
  }

  void run(SubGraphId sgId) const final {
    const auto schedule = Scheduler::vanillaComputeSchedule(m, sgId);
    using namespace poprithms::schedule;
    for (auto opId : schedule) {
      m.computeOp(opId).runSim(hts);
    }
  }
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
