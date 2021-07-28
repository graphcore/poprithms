// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef TESTUTIL_SCHEDULE_BASE_RANDOMDAG_HPP
#define TESTUTIL_SCHEDULE_BASE_RANDOMDAG_HPP

#include <vector>

namespace poprithms {
namespace schedule {
namespace baseutil {

/**
 * Create a random connected directed acyclic graph (a DAG) with N nodes.
 *
 * Algorithm proceeds as follows:
 *
 * 1) create N isolated nodes, 0...N-1.
 * 2) while not every node has a path (direct or indirect) to node N-1:
 *    add an edge i->j where i and j (0<=i<j<N) are chosen at random, and i
 *    has no path to N-1.
 *
 * How do we know this is a DAG?
 * Because 0...N-1 is one valid topological ordering, by construction.
 *
 * How do we know it's connected?
 * Because every node has a path to node N.
 * */
std::vector<std::vector<uint64_t>> randomDagConnectedToFinal(uint64_t N,
                                                             uint32_t seed);

/**
 * Create a random connected directed acyclic graph (a DAG) with N nodes.
 *
 * Algorithm proceeds as follows:
 *
 * 1) create N isolated nodes, 0...N-1.
 * 2) while not every node is connected to node 0, add a bidirectional edge
 *    between i and j, where i and j are chosen at random from [0, N) such
 *    that i != j.
 * 3) make the graph directed by replacing the birectional edges with directed
 *    edges, from the lower node index to the higher.
 *
 * How do we know this is a DAG?
 * Because 0...N-1 is one valid topological ordering, by construction step 3.
 *
 * How do we that it's connected?
 * By construction step 2, every node is connected to node 0.
 * */
std::vector<std::vector<uint64_t>> randomConnectedDag(uint64_t N,
                                                      uint32_t seed);

} // namespace baseutil
} // namespace schedule
} // namespace poprithms

#endif
