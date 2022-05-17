// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef POPRITHMS_COMMON_COMPUTE_REPLICATION_HPP
#define POPRITHMS_COMMON_COMPUTE_REPLICATION_HPP

#include <vector>

#include <poprithms/util/typedinteger.hpp>

namespace poprithms {
namespace common {
namespace compute {

/**
 * The number of replicas each tensor on an ipu device has. This corresponds
 * to the 'replication_factor' of a poplar::Graph.
 * */
class ReplicationFactor {
public:
  static ReplicationFactor create(uint64_t rf) {
    return ReplicationFactor(rf);
  }

  bool operator==(const ReplicationFactor &rhs) const { return rf == rhs.rf; }
  bool operator!=(const ReplicationFactor &rhs) const { return rf != rhs.rf; }

  bool operator<(const ReplicationFactor &rhs) const { return rf < rhs.rf; }
  bool operator<=(const ReplicationFactor &rhs) const { return rf <= rhs.rf; }

  bool operator>(const ReplicationFactor &rhs) const { return rf > rhs.rf; }
  bool operator>=(const ReplicationFactor &rhs) const { return rf >= rhs.rf; }

  uint64_t get_u64() const { return rf; }
  uint64_t get_u32() const { return rf; }
  int64_t get_i32() const { return static_cast<int32_t>(rf); }
  int64_t get_i64() const { return static_cast<int64_t>(rf); }

private:
  uint64_t rf;
  explicit ReplicationFactor(uint64_t rf_) : rf(rf_) {}
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
