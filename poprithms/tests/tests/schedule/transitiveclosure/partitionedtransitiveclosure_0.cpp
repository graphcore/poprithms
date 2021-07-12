// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <iostream>
#include <random>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/transitiveclosure/partitionedtransitiveclosure.hpp>
#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>

namespace {
using namespace poprithms::schedule::transitiveclosure;

}

void test0() {

  // 0->1   2<-3
  PartitionedTransitiveClosure ptc({{1}, {}, {}, {2}});

  if (!ptc.constrained(0, 1) || ptc.constrained(1, 2) ||
      !ptc.constrained(3, 2)) {
    throw poprithms::test::error("constraints not expected 0->1 and 3->2");
  }
}

/**
 * \param N the number of nodes
 * \param E the number of edges
 * \param seed the random seed
 *
 * A Directed Acyclic Graph with N nodes and E edges, where each edge is from
 * a random node in the planned schedule at schedule index [0, N - 10), and
 * the ending node is a distance [1, 10) beyond the start.
 *
 * */
std::vector<std::vector<uint64_t>>
randomEdges(uint64_t N, uint64_t E, uint64_t seed) {
  std::mt19937 g(seed);
  std::vector<uint64_t> schedule(N);
  std::iota(schedule.begin(), schedule.end(), 0);
  std::shuffle(schedule.begin(), schedule.end(), g);
  std::vector<std::vector<uint64_t>> edges(N);
  for (uint64_t i = 0; i < E; ++i) {
    auto start = g() % (N - 10);
    auto end   = start + 1 + g() % 9;
    edges[schedule[start]].push_back(schedule[end]);
  }
  return edges;
}

// Random tests, which check that a PartitionedTransitiveClosure and
// a TransitiveClosure provide the same results.
void test1() {

  // Generate a random graph, of varying size and edge-sparsity:
  for (uint64_t N : {23, 97, 533}) {
    for (uint64_t E : {N, 2 * N, 5 * N}) {
      for (uint64_t seed : {1011, 1012, 1013}) {
        std::mt19937 g(seed);
        auto edges = randomEdges(N, E, seed);
        PartitionedTransitiveClosure ptc(edges);
        TransitiveClosure tc(edges);

        // Assert that the the same information is provided, 2*N times:
        for (uint64_t i = 0; i < 2 * N; ++i) {
          auto start = g() % N;
          auto end   = g() % N;
          if (ptc.constrained(start, end) != tc.constrained(start, end) ||
              ptc.unconstrainedInBothDirections(start, end) !=
                  tc.unconstrainedInBothDirections(start, end)) {
            throw poprithms::test::error("ptc and tc disagree");
          }
        }
      }
    }
  }
}

void test2() {

  auto expectedComponentBitSize = [](uint64_t nOpsInComponent) {
    return 2 * //  there are 2 bitset maps, forward edges and backward edges
           nOpsInComponent * // each op in the component uses the same number
                             // of bits
           BitSetSize *      // the size of a bitset
           ((nOpsInComponent % BitSetSize != 0) +
            nOpsInComponent / BitSetSize) // the number of bitsets per op.
        ;
  };

  auto confirmSize = [](const PartitionedTransitiveClosure &ptc,
                        uint64_t expected) {
    if (ptc.nBits() != expected) {
      std::ostringstream oss;
      oss << "Expected this PartitionedTransitiveClosure to have " << expected
          << " bits, not " << ptc.nBits() << '.';
      throw poprithms::test::error(oss.str());
    }
  };

  // isolated Ops
  {
    std::vector<std::vector<uint64_t>> edges(10000);
    confirmSize(edges, edges.size() * expectedComponentBitSize(1));
  }

  {
    // 0 -> 1 -> ... -> 999.
    std::vector<std::vector<uint64_t>> edges(1000);
    for (uint64_t i = 0; i < 999; ++i) {
      edges[i] = {i + 1};
    }
    confirmSize(edges, 1 * expectedComponentBitSize(1000));
  }

  {
    // 0 -> ... -> 99  100 -> ... -> 199 ... 900 -> 999.
    std::vector<std::vector<uint64_t>> edges(1000);
    for (uint64_t i = 0; i < 999; ++i) {
      if (i % 100 != 99) {
        edges[i] = {i + 1};
      }
    }
    confirmSize(edges, 10 * expectedComponentBitSize(100));
  }

  {
    // 0 -> 9 10 -> 99  100 -> 999
    std::vector<std::vector<uint64_t>> edges(1000);
    for (uint64_t i = 0; i < 999; ++i) {
      if (i != 9 && i != 99) {
        edges[i] = {i + 1};
      }
    }
    confirmSize(edges,
                expectedComponentBitSize(10) + expectedComponentBitSize(90) +
                    expectedComponentBitSize(900));
  }
}

int main() {

  test0();
  test1();
  test2();
  return 0;
}
