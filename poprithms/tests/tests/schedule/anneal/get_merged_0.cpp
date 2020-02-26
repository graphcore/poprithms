
#include <algorithm>
#include <iostream>
#include <vector>
#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>

namespace {
using namespace poprithms::schedule::anneal;

void test0() {

  Graph g;
  auto ops = g.insertOps({"op0", "op1", "op2", "op3", "op4", "op5"});
  //
  //   0
  // 1   2
  // 3   4
  //   5
  //
  g.insertConstraint(ops[0], ops[1]);
  g.insertConstraint(ops[0], ops[2]);
  g.insertConstraint(ops[1], ops[3]);
  g.insertConstraint(ops[2], ops[4]);
  g.insertConstraint(ops[3], ops[5]);
  g.insertConstraint(ops[4], ops[5]);

  auto merged0 = g.getMerged({{1, 3}, {2, 4}});
  auto g0      = std::get<0>(merged0);
  g0.initialize();
  if (g0.nOps() != 4) {
    throw error("Expected 4 Ops in merged Graph");
  }

  auto linkMerged0 = g.getLinkMerged();
  if (std::get<0>(linkMerged0).nOps() != g.nOps()) {
    throw error(
        "link merge with no links should result in graph of same size");
  }
}

void test1() {

  //
  //   0------5
  //  / \     |
  // 1   2    6
  // |   |    |
  // |   3    7
  //  \ /     |
  //   4      8
  //    \     |
  //     9    10
  //      \  /
  //       11

  Graph g;
  for (uint64_t i = 0; i < 12; ++i) {
    auto op = g.insertOp("Op" + std::to_string(i));
  }
  for (OpAddress i : {2, 3, 5, 6, 7, 10}) {
    g.insertConstraint(i, i + 1);
  }
  g.insertConstraint(0, 2);
  g.insertConstraint(0, 5);
  g.insertConstraint(1, 4);
  g.insertConstraint(4, 9);
  g.insertConstraint(9, 11);
  auto merged0 = g.getTightMerged();
  auto parents = std::get<1>(merged0);
  for (auto &p : parents) {
    std::sort(p.begin(), p.end());
  }
  std::sort(parents.begin(), parents.end());
  if (parents != Graph::ParentGraphOps{
                     {0}, {1}, {2, 3}, {4, 9}, {5, 6, 7, 8}, {10}, {11}}) {
    throw error("unexpected mapping generated in getTightMerged");
  }
}

} // namespace

int main() {
  test0();
  test1();
  return 0;
}
