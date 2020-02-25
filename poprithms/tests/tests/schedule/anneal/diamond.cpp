#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>
#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>
#include <testutil/schedule/anneal/annealcommandlineoptions.hpp>

int main(int argc, char **argv) {

  using namespace poprithms::schedule::anneal;
  using namespace poprithms::schedule::anneal;
  auto opts = AnnealCommandLineOptions().getCommandLineOptionsMap(
      argc, argv, {"N"}, {"The number of intermediate Ops in the diamond"});
  auto N = std::stoi(opts.at("N"));

  Graph graph;

  //            x
  //           / \
  //      x x x x x x x (the N intermediate Ops)
  //           \ /
  //            x

  auto root = graph.insertOp("root");
  auto tail = graph.insertOp("tail");

  for (int i = 0; i < N; ++i) {
    // weight decreases in N, so we expect ops with low addresses (heavy
    // weights) to be scheduled first
    double w0 = N + 1 - i;
    double w1 = 5;
    auto a0   = graph.insertAlloc(w0);
    auto a1   = graph.insertAlloc(w1);
    auto op   = graph.insertOp("op" + std::to_string(i));
    graph.insertOpAlloc(op, a0);
    graph.insertOpAlloc(op, a1);
    graph.insertOpAlloc(root, a0);
    graph.insertOpAlloc(tail, a1);
    graph.insertConstraint(root, op);
    graph.insertConstraint(op, tail);
  }

  graph.initialize(KahnTieBreaker::RANDOM);

  graph.minSumLivenessAnneal({});
  std::vector<OpAddress> expected;
  expected.push_back(root);
  for (int i = 0; i < N; ++i) {
    expected.push_back(i + 2);
  }
  expected.push_back(tail);

  for (ScheduleIndex i = 0; i < graph.nOps(); ++i) {
    if (graph.scheduleToOp(i) != expected[i]) {
      throw error("unexpected schedule");
    }
  }
}
