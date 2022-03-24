// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_VANILLA_VANILLA_MAP_HPP
#define POPRITHMS_SCHEDULE_VANILLA_VANILLA_MAP_HPP

#include <array>
#include <map>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <poprithms/schedule/vanilla/vanilla.hpp>

namespace poprithms {
namespace schedule {
namespace vanilla {

/**
 * Obtain a schedule from an edge map provided as a Map. key:values pairs in
 * the Map denote forward edges, so keys will always appear before values.
 * */
template <typename Map>
std::vector<typename Map::key_type>
getSchedule(const Map &fwdEdgesSparse, ErrorIfCycle eic, VerifyEdges ve) {
  using T = typename Map::key_type;

  // set of all T's observed in the Map, both as keys ('from's) and values
  // ('to's). Using set and not unordered_set to ensure order and
  // reproducibility across platforms.
  std::set<T> tSet;
  for (const auto &k_vs : fwdEdgesSparse) {
    tSet.insert(k_vs.first);
    for (const auto &v : k_vs.second) {
      tSet.insert(v);
    }
  }

  const auto N = tSet.size();
  std::vector<T> fromCompact(tSet.cbegin(), tSet.cend());
  std::unordered_map<T, uint64_t> toCompact;
  for (uint64_t i = 0; i < N; ++i) {
    toCompact.insert({fromCompact[i], i});
  }

  std::vector<std::vector<uint64_t>> fwdEdgesCompact(N);
  for (const auto &k_vs : fwdEdgesSparse) {
    auto &from = fwdEdgesCompact[toCompact[k_vs.first]];
    for (auto v : k_vs.second) {
      from.push_back(toCompact[v]);
    }
  }

  auto compactSched = getSchedule_u64(fwdEdgesCompact, eic, ve);
  std::vector<T> sched;
  for (auto c : compactSched) {
    sched.push_back(fromCompact[c]);
  }

  return sched;
}

} // namespace vanilla
} // namespace schedule
} // namespace poprithms

#endif
