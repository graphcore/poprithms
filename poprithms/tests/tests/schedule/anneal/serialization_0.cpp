#include <iostream>
#include <limits>
#include <string>

#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>

namespace {

using namespace poprithms::schedule::anneal;

bool serializesCorrectly(const Graph &g) {
  auto serialization = g.getSerializationString();
  auto g2            = Graph::fromSerializationString(serialization);
  return g2 == g && g2.getSerializationString() == g.getSerializationString();
}

void test0() {
  Graph g;
  if (!serializesCorrectly(g)) {
    throw error("Serialization failed for empty Graph");
  }
}

void test1() {
  Graph g;
  g.insertLink(g.insertOp("op1"), g.insertOp("op2"));
  if (!serializesCorrectly(g)) {
    throw error("Serialization failed for Graph with no Allocs");
  }
}

void test2() {
  Graph g;
  g.insertAlloc(100.0);
  g.insertAlloc(2.0);
  if (!serializesCorrectly(g)) {
    throw error("Serialization failed for Graph with no Ops");
  }
}

void test3() {
  auto g   = Graph();
  auto op0 = g.insertOp("op0");
  auto op1 = g.insertOp("op1");
  auto op2 = g.insertOp("op2");
  auto op3 = g.insertOp("op3");
  auto op4 = g.insertOp("operator_four  [[[((({{{ \" \\ ");
  auto op5 = g.insertOp("operator_five");
  auto op6 = g.insertOp("operator_six");
  g.insertConstraint(op0, op1);
  g.insertConstraint(op0, op2);
  g.insertConstraint(op1, op3);
  g.insertConstraint(op2, op3);
  g.insertLink(op5, op6);

  // insert some common and unusual values, and verify that double
  // serialization is lossless;
  auto alloc0 = g.insertAlloc(123.0);
  auto alloc1 = g.insertAlloc(AllocWeight::numericMaxLimit());
  auto alloc2 = g.insertAlloc(std::numeric_limits<double>::lowest());
  auto alloc3 = g.insertAlloc(std::numeric_limits<double>::min());
  auto alloc4 = g.insertAlloc(std::numeric_limits<double>::max());

  //              .
  // 7.7777... (7.7)
  std::vector<char> sevens(100, '7');
  sevens[1]      = '.';
  double dSevens = std::stod(std::string{sevens.cbegin(), sevens.cend()});

  auto alloc5 = g.insertAlloc(dSevens);
  auto alloc6 = g.insertAlloc(AllocWeight(dSevens * 1e-19, -1));
  auto alloc7 = g.insertAlloc(-1.0);
  auto alloc8 = g.insertAlloc(-0.0);

  g.insertOpAlloc({op0, op1}, alloc0);
  g.insertOpAlloc(op1, alloc1);
  g.insertOpAlloc(op1, alloc3);
  g.insertOpAlloc(op1, alloc8);

  g.finalize();
  auto newGraph = Graph::fromSerializationString(g.getSerializationString());

  if (newGraph != g) {
    std::cout << newGraph << std::endl;
    std::cout << g << std::endl;
    throw error(
        "Graph and serialized Graph differ (direct Graph comparison)");
  }
  if (newGraph.getSerializationString() != g.getSerializationString()) {
    throw error(
        "Graph and serialiazed Graph differ (serialization comparison)");
  }
}
} // namespace

int main() {
  test0();
  test1();
  test2();
  test3();
  return 0;
}
