#include <algorithm>
#include <iostream>
#include <numeric>
#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>

namespace {
poprithms::schedule::anneal::Graph getBaseGraph(uint64_t N) {

  // StartOp____0
  // | | | |    |
  // 1 2 3 4...N-2
  // | | | |    |
  // FinalOp___N-1
  //
  // allocations : 0        :  none.
  //               1 -> N-2 :  of size i**2 a.k.a. i^2 a.k.a. i*i
  //               N-1      :  all allocations of 1->N-2

  assert(N % 2 == 0);

  using namespace poprithms::schedule::anneal;

  Graph g;
  std::vector<AllocAddress> allocIds;
  for (uint64_t i = 0; i < N; ++i) {
    auto opId = g.insertOp("op_" + std::to_string(i));
    if (i == 0) {
    } else if (i == N - 1) {
      for (auto allocId : allocIds) {
        g.insertOpAlloc(opId, allocId);
      }
      for (uint64_t j = 0; j < i; ++j) {
        g.insertConstraint(j, i);
      }
    } else {
      allocIds.push_back(g.insertAlloc(static_cast<AllocWeight>(i * i)));
      g.insertConstraint(0, i);
      g.insertOpAlloc(opId, allocIds.back());
    }
  }
  return g;
}
} // namespace

int main() {

  using namespace poprithms::schedule::anneal;
  using namespace poprithms;

  constexpr uint64_t N = 10;
  std::vector<OpAddress> allIds(N);
  std::iota(allIds.begin(), allIds.end(), 0);

  // get a schedule with proxy attraction Ops removed.
  auto getFilteredSchedule = [N](const Graph &g) {
    std::vector<std::tuple<ScheduleIndex, OpAddress>> filtered;
    // The first N Ops are always the N true Ops:
    for (OpAddress a = 0; a < N; ++a) {
      filtered.push_back({g.opToSchedule(a), a});
    }
    std::sort(filtered.begin(), filtered.end());
    std::vector<OpAddress> subSchedule;
    for (auto x : filtered) {
      subSchedule.push_back(std::get<1>(x));
    }
    return subSchedule;
  };

  // Test 1 : no additional priorities added.
  auto g = getBaseGraph(N);
  g.initialize();
  g.minSumLivenessAnneal();
  std::vector<OpAddress> expected(N);
  std::iota(expected.begin(), expected.end(), 0);
  auto scheduled = getFilteredSchedule(g);
  for (ScheduleIndex i = 0; i < N; ++i) {
    if (scheduled[i] != expected[i]) {
      throw error("Failure in Test 1, schedule not as expected");
    }
  }

  // Test 2 : add sub-priorities proportional to ID : should be unchanged
  g = getBaseGraph(N);
  g.insertStartAttractors(allIds, allIds, +1);
  g.initialize();
  g.minSumLivenessAnneal();
  scheduled = getFilteredSchedule(g);
  for (ScheduleIndex i = 0; i < N; ++i) {
    if (scheduled[i] != expected[i]) {
      throw error("Failure in Test 2, schedule not as expected");
    }
  }

  // Test 3 : add super-priorities proportional to ID. These should now
  // reverse the order of the schedule, which should now follow the priorities
  g = getBaseGraph(N);
  g.insertStartAttractors(allIds, allIds, -1);
  g.initialize();
  g.minSumLivenessAnneal();
  scheduled = getFilteredSchedule(g);
  // for N = 6: 0 4 3 2 1 5
  std::iota(expected.rbegin(), expected.rend(), 0);
  expected[0]     = 0;
  expected.back() = N - 1;
  for (ScheduleIndex i = 0; i < N; ++i) {
    if (scheduled[i] != expected[i]) {
      throw error("Failure in Test 3, schedule not as expected");
    }
  }

  // Test 4 : attractions between Ops (ala popart's tied Ops)
  g = getBaseGraph(N);
  std::vector<std::array<OpAddress, 2>> partners;
  for (OpAddress i = 0; i < N / 2; ++i) {
    partners.push_back({i, N - 1 - i});
  }
  g.insertAttractions(partners, {1.0, -1});
  // we have inserted the following super-strong attractions:
  //     0 >-< N-1
  //     1 >-< N-2
  //     2 >-< N-3
  //     3 >-< N-4
  //        .
  //        .
  // N/2-1 >-< N/2
  //
  // the priority is to get these partners as close together as possible.
  //
  // The secondary consideration is the actual allocations, which are
  // quadratic in OpAddress. This secondary consideration encourages pairs
  // with larger difference in OpAddresses to appear later in the schedule
  // (for example, 1**2 + 9**2 > 4**2 + 5**2). Thus we expect:
  //
  // 0 4 5 3 6 2 7 1 8 9
  // - === === === === -
  //
  // The first and last Ops cannot be matched together in the schedule due to
  // topological constraints.
  expected[0] = 0;
  for (uint64_t i = 1; i < N / 2; ++i) {
    expected[N - 1 - 2 * i] = i;
    expected[N - 2 * i]     = N - i - 1;
  }
  expected.back() = N - 1;
  g.initialize();
  g.minSumLivenessAnneal();
  scheduled = getFilteredSchedule(g);
  for (ScheduleIndex i = 0; i < N; ++i) {
    if (scheduled[i] != expected[i]) {
      throw error("Failure in Test 4, schedule not as expected");
    }
  }

  // Test 5: in addition to attractions between Ops, and super-priorities. We
  // now expect 0 5 4 6 3 7 2 8 1 9
  // - === === === === -
  g = getBaseGraph(N);
  g.insertStartAttractors(allIds, allIds, -1);
  g.insertAttractions(partners, {1.0, -2});
  for (uint64_t i = 1; i < N / 2; ++i) {
    std::swap(expected[2 * i - 1], expected[2 * i]);
  }
  g.initialize();
  g.minSumLivenessAnneal();
  scheduled = getFilteredSchedule(g);
  for (ScheduleIndex i = 0; i < N; ++i) {
    if (scheduled[i] != expected[i]) {
      throw error("Failure in Test 5, schedule not as expected");
    }
  }

  return 0;
}
