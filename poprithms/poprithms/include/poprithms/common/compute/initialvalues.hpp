// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_INITIALVALUES_HPP
#define POPRITHMS_COMMON_COMPUTE_INITIALVALUES_HPP

#include <map>
#include <vector>

#include <poprithms/common/compute/hosttensor.hpp>
#include <poprithms/common/multiout/tensorid.hpp>

namespace poprithms {
namespace common {
namespace compute {

using poprithms::common::multiout::ContiguousOutIndexSubset;
using poprithms::common::multiout::OutIndex;

/**
 * Optional initial values of an op's outputs, with each replica optionally
 * having a different value.
 * */
class InitialValues {

public:
  InitialValues() = delete;
  /**
   * The number of output tensors of the op. No initial values are set -- to
   * set values use the method #setValue.
   * */
  InitialValues(uint64_t nOutTensors) : chts(nOutTensors) {}

  InitialValues(const InitialValues &rhs) = default;
  InitialValues(InitialValues &&rhs)      = default;

  InitialValues &operator=(const InitialValues &) = default;
  InitialValues &operator=(InitialValues &&)      = default;

  /**
   * Set the initial value of output #o for replica #replica to #initVal.
   * */
  void setValue(OutIndex o, uint64_t replica, const HostTensor &initVal);

  /**
   * Numerical comparison of initial values.
   * */
  bool operator==(const InitialValues &rhs) const { return chts == rhs.chts; }
  bool operator!=(const InitialValues &rhs) const { return !operator==(rhs); }

  void reduce(const ContiguousOutIndexSubset &coin) { coin.reduce(chts); }

  std::map<uint64_t, HostTensor> getInitialValues(OutIndex o) const;

  uint64_t nOutTensors() const { return chts.size(); }

private:
  std::vector<std::map<uint64_t, ComparableHostTensor>> chts;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
