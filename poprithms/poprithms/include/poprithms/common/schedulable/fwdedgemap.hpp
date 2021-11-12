// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_SCHEDULABLE_FWDEDGEMAP_HPP
#define POPRITHMS_COMMON_SCHEDULABLE_FWDEDGEMAP_HPP

#include <unordered_map>

#include <poprithms/common/multiout/opid.hpp>
namespace poprithms {
namespace common {
namespace schedulable {

using common::multiout::OpId;
using common::multiout::OpIds;
/**
 * Represent a subset of a graph's ops and the dependencies between them. A
 * graph's ops' ids may be a discontiguous set of integers (like {0,4,5}).
 * This class stores a mapping between the ids and a contiguous range of
 * integers (like {0,1,2}). This
 *
 * (1) makes it possible to use schedulers which expect a contiguous range,
 * (2) make it more efficient to perform certain operations.
 * */
class FwdEdgeMap {

public:
  /**
   * Initialize the edge map from a set of distinct OpIds, but without any
   * edges. Edges are added with 'insertEdge'.
   * */
  FwdEdgeMap(const OpIds &);

  /**
   * Map a set of ids in the compact/contiguous back to the original set of
   * non-contiguous OpIds.
   * */
  OpIds unpacked(const std::vector<uint64_t> &s_u64) const;

  /**
   * Insert an edge between 2 ops.
   * */
  void insertEdge(OpId from, OpId to) {
    fwdEdgesCompact_[toCompact_[from]].push_back(toCompact_[to]);
  }

  /**
   * Reserve memory in the vector storing the outwards edges of op #id. This
   * is used for more efficient incremental growing of the edge map.
   * */
  void reserve(OpId id, uint64_t n) {
    fwdEdgesCompact_[toCompact_[id]].reserve(n);
  }

  void append(std::ostream &) const;

  const std::vector<std::vector<uint64_t>> &fwdEdgesCompact() const {
    return fwdEdgesCompact_;
  }

private:
  // A map from original (non-contiguous) ids to the compact (contiguous)
  // ids.
  std::unordered_map<OpId, uint64_t> toCompact_;

  // The forward edges of the compact representation.
  std::vector<std::vector<uint64_t>> fwdEdgesCompact_;

  // A mapping to the original OpIds.
  OpIds fromCompact_;
};

std::ostream &operator<<(std::ostream &, const FwdEdgeMap &);

} // namespace schedulable
} // namespace common
} // namespace poprithms

#endif
