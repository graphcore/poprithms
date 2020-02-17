#include <iostream>
#include <string>
#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>

int main() {
  using namespace poprithms::schedule::anneal;

  Graph g;

  uint64_t nOps = 5;
  for (uint64_t i = 0; i < nOps; ++i) {
    g.insertOp("Op" + std::to_string(i));
  }

  for (uint64_t i = 0; i < nOps - 1; ++i) {
    g.insertConstraint(i, nOps - 1);
  }

  auto g2 = g;
  for (uint64_t i = 0; i < nOps; ++i) {
    if (g2.getOp(i) != g.getOp(i)) {
      throw error("Expect Ops in copied Graph to compare to equal");
    }
  }

  for (uint64_t i = 0; i < nOps - 1; ++i) {
    if (!g.getOp(i).hasOut(nOps - 1) || !g.getOp(nOps - 1).hasIn(i)) {
      throw error("Unexpected in/out");
    }
  }

  Op op0(1000, "standaloneOp");
  op0.insertIn(1);
  op0.insertIn(3);
  op0.insertIn(2);
  op0.insertIn(4);
  if (!op0.hasIn(2)) {
    throw error("2 in an input to op0");
  }
  op0.removeIn(2);
  if (op0.hasIn(2)) {
    throw error("2 has been removed as an input to op0");
  }

  return 0;
}
