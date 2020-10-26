// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <poprithms/schedule/vanilla/error.hpp>
#include <poprithms/schedule/vanilla/vanilla.hpp>

namespace poprithms {
namespace schedule {
namespace vanilla {

namespace {

template <typename T> bool valid(T x, uint64_t end) {
  return x < static_cast<T>(end);
}

template <> bool valid(int64_t x, uint64_t end) {
  return x >= 0 && static_cast<uint64_t>(x) < end;
}

// Kahn's algorithm
// https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
template <typename T>
std::vector<T> kahn(const std::vector<std::vector<T>> &fwdEdges,
                    ErrorIfCycle eic,
                    VerifyEdges ve) {

  const auto N = fwdEdges.size();

  if (ve == VerifyEdges::Yes) {
    for (uint64_t start = 0; start < N; ++start) {
      for (auto end : fwdEdges[start]) {
        if (!valid<T>(end, N)) {
          std::ostringstream oss;
          oss << "Invalid edge (" << start << "->" << end
              << ") in graph with " << N << " nodes. ";
          throw error(oss.str());
        }
      }
    }
  }

  // Count the number of dependencies each Op has
  std::vector<uint64_t> nOutstandingDeps(N, 0);
  for (uint64_t from = 0; from < N; ++from) {
    for (const auto to : fwdEdges[from]) {
      ++nOutstandingDeps[to];
    }
  }

  // Get the Ops which have no dependencies: they're ready to go into the
  // schedule.
  std::vector<T> readyToSchedule;
  for (uint64_t i = 0; i < N; ++i) {
    if (nOutstandingDeps[i] == 0) {
      readyToSchedule.push_back(i);
    }
  }

  std::vector<T> schedule;
  schedule.reserve(N);
  while (!readyToSchedule.empty()) {
    const auto nxt = readyToSchedule.back();
    readyToSchedule.pop_back();
    for (const auto to : fwdEdges[nxt]) {
      --nOutstandingDeps[to];
      if (nOutstandingDeps[to] == 0) {
        readyToSchedule.push_back(to);
      }
    }
    schedule.push_back(nxt);
  }

  if (eic == ErrorIfCycle::Yes && schedule.size() != N) {
    std::ostringstream oss;
    oss << "Only " << schedule.size() << " nodes of " << N
        << " scheduled, there is a cycle in the Graph. ";
    throw error(oss.str());
  }
  return schedule;
}
} // namespace

std::vector<uint64_t>
getSchedule_u64(const std::vector<std::vector<uint64_t>> &fwdEdges,
                ErrorIfCycle eic,
                VerifyEdges ve) {
  return kahn<uint64_t>(fwdEdges, eic, ve);
}

std::vector<int64_t>
getSchedule_i64(const std::vector<std::vector<int64_t>> &fwdEdges,
                ErrorIfCycle eic,
                VerifyEdges ve) {
  return kahn<int64_t>(fwdEdges, eic, ve);
}

} // namespace vanilla
} // namespace schedule
} // namespace poprithms
