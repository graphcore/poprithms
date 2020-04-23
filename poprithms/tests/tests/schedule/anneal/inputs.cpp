#include <iostream>

#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>

int main() {

  using namespace poprithms::schedule::anneal;

  Graph g;
  auto op0 = g.insertOp("op0");
  auto op1 = g.insertOp("op1");
  auto op2 = g.insertOp("op2");
  auto op3 = g.insertOp("op3");
  g.insertConstraint(op0, op1);
  g.insertConstraint(op0, op3);
  g.insertConstraint(op2, op1);
  g.insertConstraint(op2, op3);

  std::vector<OpAddress> expected = {0, 2};
  auto inputs                     = g.getInputOps();
  std::sort(inputs.begin(), inputs.end());
  if (inputs != expected) {
    throw error("incorrect input Ops");
  }
  return 0;
}
