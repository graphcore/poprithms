// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_MULTIOUT_FWDEDGEMAP_HPP
#define POPRITHMS_COMMON_MULTIOUT_FWDEDGEMAP_HPP

#include <map>
#include <unordered_map>

#include <poprithms/common/multiout/opid.hpp>

namespace poprithms {
namespace common {
namespace multiout {

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
  explicit FwdEdgeMap(const OpIds &);

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

  /**
   * The reverse edges of the forward edge map.
   * */
  std::vector<std::vector<uint64_t>> createBwdEdgesCompact() const;

  OpIds outs(OpId opId) const {
    return unpacked(fwdEdgesCompact().at(compactId(opId)));
  }

  uint64_t nOps() const { return fwdEdgesCompact_.size(); }

  uint64_t compactId(OpId opId) const { return toCompact_.at(opId); }

  OpId opId(uint64_t compactId) const { return fromCompact_[compactId]; }

  /**
   * Convert a map with OpId keys to a map with the compact mappings of the
   * OpIds.
   * */
  template <typename X>
  std::map<uint64_t, X> getCompact(const std::map<OpId, X> &sparse) const {
    std::map<uint64_t, X> dense;
    for (const auto &iter : sparse) {
      dense.insert({compactId(iter.first), iter.second});
    }
    return dense;
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

} // namespace multiout
} // namespace common
} // namespace poprithms

#endif
