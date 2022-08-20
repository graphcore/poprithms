// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_SIMTENSORMAP_HPP
#define POPRITHMS_COMMON_COMPUTE_SIMTENSORMAP_HPP

#include <memory>
#include <unordered_map>

#include <poprithms/common/multiout/tensormap.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/util/circularcounter.hpp>
#include <poprithms/util/copybyclone.hpp>

namespace poprithms {
namespace common {
namespace compute {

using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OpIds;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;

using HostTensors = poprithms::compute::host::Tensors;
using HostTensor  = poprithms::compute::host::Tensor;

/**
 * A class for storing tensors and certain op states of a
 * common::compute::Graph, in host memory. It can be used to numerically
 * simulate a replicated graph.
 * */
class SimTensorMap
    : public poprithms::common::multiout::TensorMap<HostTensors> {
public:
  SimTensorMap()                   = default;
  virtual ~SimTensorMap() override = default;

  /**
   * The number of compute::host::Tensors stored for each of the tensors in
   * #tIds, if they are the same.
   * */

  uint64_t getNTensorsByUnanimity(const TensorIds &tIds) const;

  /**
   * \return A vector of HostTensors, taken at index #index for each of the
   *         tensors in #tIds (element #i is getValue(tIds[i]).at(index)).
   * */
  HostTensors getTensors(const TensorIds &tIds, uint64_t index) const;

  /**
   * Copy the values of the host tensor(s) of #src to the host tensor(s) for
   * #dst.
   * */
  void copy(const TensorId &src, const TensorId &dst) const;

  /**
   * For all i copy srcs[i] to dsts[i], where #srcs and #dsts must be the
   * same.
   * */
  void copy(const TensorIds &srcs, const TensorIds &dsts) const;

  /**
   * Clone this SimTensorMap. Host tensors are shallow copied.
   * */
  std::unique_ptr<SimTensorMap> clone() const;

  /**
   * Insert a counter for an op #opId, where op #opId has some state which
   * is incremented with modular arithmetic. An example is a copy to/from a
   * circular buffer, where the src/dst pointer increments and then wraps
   * around at the end.
   * */
  void insertCounter(OpId opId, uint64_t modulus_) {
    counters.insert(opId, modulus_);
  }

  uint64_t getCounterState(OpId opId) const { return counters.state(opId); }

  void incrementCounter(OpId opId) { counters.increment(opId); }

private:
  poprithms::util::CircularCounters<OpId> counters;

  virtual void noWeakVTables();
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
