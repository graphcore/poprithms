// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "error.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <sstream>

#include <poprithms/schedule/vanilla/pathcount.hpp>
#include <poprithms/schedule/vanilla/types.hpp>
#include <poprithms/schedule/vanilla/vanilla.hpp>

namespace poprithms {
namespace schedule {
namespace vanilla {

std::ostream &operator<<(std::ostream &ost, CountType ct) {
  switch (ct) {
  case CountType::Add: {
    ost << "Add";
    return ost;
  }
  case CountType::Max: {
    ost << "Max";
    return ost;
  }
  case CountType::Min: {
    ost << "Min";
    return ost;
  }
  }

  throw error("Unrecognised CountType");
}

template <typename Helper>
std::vector<uint64_t> tPathCount(const Edges<uint64_t> &fwdEdges,
                                 ErrorIfCycle eic,
                                 VerifyEdges ves) {

  // forward schedule.
  auto sc = getSchedule_u64(fwdEdges, eic, ves);

  // number of nodes in the graph.
  const auto N = fwdEdges.size();

  // going through the nodes in reverse schedule order, set the count of each
  // node based on the downstream values.
  std::vector<uint64_t> counts(N);
  std::vector<uint64_t> downStreams;
  for (uint64_t i = 0; i < N; ++i) {
    const auto node = sc[N - i - 1];
    downStreams.resize(fwdEdges[node].size());
    for (uint64_t j = 0; j < fwdEdges[node].size(); ++j) {
      downStreams[j] = counts[fwdEdges[node][j]];
    }
    counts[node] = Helper::get(downStreams);
  }
  return counts;
}

class CountAdd {
public:
  static uint64_t get(const std::vector<uint64_t> &downStreams) {
    if (downStreams.size() == 0) {
      return 1;
    }
    return std::accumulate(downStreams.cbegin(), downStreams.cend(), 0ULL);
  }
};

class CountMax {
public:
  static uint64_t get(const std::vector<uint64_t> &vs) {
    return 1 +
           std::accumulate(vs.cbegin(), vs.cend(), 0ull, [](auto a, auto b) {
             return std::max<uint64_t>(a, b);
           });
  }
};

class CountMin {
public:
  static uint64_t get(const std::vector<uint64_t> &vs) {
    if (vs.size() == 0) {
      return 1;
    }
    return 1 + std::accumulate(
                   vs.cbegin(),
                   vs.cend(),
                   std::numeric_limits<uint64_t>::max(),
                   [](auto a, auto b) { return std::min<uint64_t>(a, b); });
  }
};

// template <typename Helper>
std::vector<uint64_t> PathCounter::count(const Edges<uint64_t> &fwdEdges,
                                         CountType t,
                                         ErrorIfCycle eic,
                                         VerifyEdges ves) {

  switch (t) {
  case CountType::Add: {
    return tPathCount<CountAdd>(fwdEdges, eic, ves);
  }
  case CountType::Max: {
    return tPathCount<CountMax>(fwdEdges, eic, ves);
  }
  case CountType::Min: {
    return tPathCount<CountMin>(fwdEdges, eic, ves);
  }
  }

  throw error("Unrecognised CountType");
}

} // namespace vanilla
} // namespace schedule
} // namespace poprithms
