// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iterator>
#include <numeric>
#include <random>
#include <set>
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/vanilla/vanilla.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

template <class T>
std::ostream &operator<<(std::ostream &ost, const std::vector<T> &ts) {
  poprithms::util::append(ost, ts);
  return ost;
}

using namespace poprithms::schedule::vanilla;

using Node      = uint64_t;
using Priority  = double;
using AllocSize = int;

void assertSchedule(const Edges<Node> &edges,
                    const Priorities<Node, Priority> &priorities,
                    const std::vector<AllocSize> &allocSizes,
                    const std::vector<std::vector<Node>> &allocsToNodes,
                    const std::vector<Node> &expected) {

  Links<Node> links{};
  auto observed =
      GreedyScheduler<Node, Priority, AllocSize>::kahn(edges,
                                                       priorities,
                                                       links,
                                                       allocSizes,
                                                       allocsToNodes,
                                                       ErrorIfCycle::Yes,
                                                       VerifyEdges::Yes);

  if (observed != expected) {
    std::ostringstream oss;
    oss << "Expected the schedule " << expected << " but observed "
        << observed << ". ";
    oss << "This with " << priorities.size() << " priorities and "
        << allocSizes.size() << " allocs. ";
    throw poprithms::test::error(oss.str());
  }
}

void test0() {

  //
  //
  //         0
  //         |
  //      +--+--+
  //      |     |
  //      1--+--2
  //         |
  //         3

  Edges<Node> edges{{1, 2}, {3}, {3}, {}};

  AllocSize a01{100};
  AllocSize a02{1};
  AllocSize a13{1};
  AllocSize a23{1};
  std::vector<AllocSize> allocSizes{a01, a02, a13, a23};

  Edges<Node> allocsToNodes{{0, 1}, {0, 2}, {1, 3}, {2, 3}};

  // assert that priority trumps non-priority.
  Priorities<Node, Priority> priorities{{1, 2.0}, {2, 1.0}};
  assertSchedule(edges, priorities, allocSizes, allocsToNodes, {0, 1, 2, 3});
  priorities = {{2, 2.0}, {1, 1.0}};
  assertSchedule(edges, priorities, allocSizes, allocsToNodes, {0, 2, 1, 3});

  // when equal priorities, the allocs are used
  for (Priorities<Node, Priority> ps :
       std::vector<Priorities<Node, Priority>>{
           {{}, {{1, 100.}, {2, 100.}}}}) {

    allocsToNodes = {{0, 1}, {0, 2}, {1, 3}, {2, 3}};
    allocSizes    = {100, 1, 1, 1};
    assertSchedule(edges, ps, allocSizes, allocsToNodes, {0, 1, 2, 3});

    allocSizes = {100, 1, 200, 1};
    assertSchedule(edges, ps, allocSizes, allocsToNodes, {0, 2, 1, 3});

    allocSizes = {100, 300, 200, 350};
    assertSchedule(edges, ps, allocSizes, allocsToNodes, {0, 2, 1, 3});

    allocSizes = {100, 300, 200, 3050};
    assertSchedule(edges, ps, allocSizes, allocsToNodes, {0, 1, 2, 3});
  }
}
} // namespace

int main() {
  test0();
  return 0;
}
