// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <schedule/vanilla/error.hpp>
#include <sstream>

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

template <typename T>
void verifyEdges(const std::vector<std::vector<T>> &fwdEdges) {
  const auto N = fwdEdges.size();
  for (uint64_t start = 0; start < N; ++start) {
    for (auto end : fwdEdges[start]) {
      if (!valid<T>(end, N)) {
        std::ostringstream oss;
        oss << "Invalid edge (" << start << "->" << end << ") in graph with "
            << N << " nodes. ";
        throw error(oss.str());
      }
    }
  }
}

template <typename T>
std::vector<uint64_t>
getOutstandingCount(const std::vector<std::vector<T>> &fwdEdges) {

  const auto N = fwdEdges.size();

  // Count the number of dependencies each Op has
  std::vector<uint64_t> nOutstandingDeps(N, 0);
  for (uint64_t from = 0; from < N; ++from) {
    for (const auto to : fwdEdges[from]) {
      ++nOutstandingDeps[to];
    }
  }
  return nOutstandingDeps;
}

// Kahn's algorithm
// https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
template <typename T>
std::vector<T> kahn(const std::vector<std::vector<T>> &fwdEdges,
                    ErrorIfCycle eic,
                    VerifyEdges ve) {

  if (ve == VerifyEdges::Yes) {
    verifyEdges(fwdEdges);
  }

  const auto N          = fwdEdges.size();
  auto nOutstandingDeps = getOutstandingCount(fwdEdges);

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

template <typename T>
bool getHasUniqueSchedule(const Edges<T> &fwdEdges, VerifyEdges ve) {
  if (ve == VerifyEdges::Yes) {
    verifyEdges(fwdEdges);
  }

  const auto N          = fwdEdges.size();
  auto nOutstandingDeps = getOutstandingCount(fwdEdges);

  uint64_t nScheduled{0};

  std::vector<T> schedulable;
  for (uint64_t i = 0; i < N; ++i) {
    if (nOutstandingDeps[i] == 0) {
      schedulable.push_back(i);
    }
  }

  while (schedulable.size() == 1) {
    const auto nxt = schedulable.back();
    schedulable.pop_back();
    ++nScheduled;
    for (const auto to : fwdEdges[nxt]) {
      --nOutstandingDeps[to];
      if (nOutstandingDeps[to] == 0) {
        schedulable.push_back(to);
      }
    }
  }

  if (nScheduled == N) {
    return true;
  }
  return false;
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

bool hasUniqueSchedule_u64(const Edges<uint64_t> &fwdEdges, VerifyEdges ve) {
  return getHasUniqueSchedule<uint64_t>(fwdEdges, ve);
}
bool hasUniqueSchedule_i64(const Edges<int64_t> &fwdEdges, VerifyEdges ve) {
  return getHasUniqueSchedule<int64_t>(fwdEdges, ve);
}

} // namespace vanilla
} // namespace schedule
} // namespace poprithms
