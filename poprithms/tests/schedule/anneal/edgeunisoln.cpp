#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>
#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>

int main(int argc, char **argv) {

  using namespace poprithms::schedule::anneal;

  Graph graph;

  // X X X (no constraints)
  auto a = graph.insertOp("a");
  auto b = graph.insertOp("b");
  auto c = graph.insertOp("c");
  graph.finalize();

  if (graph.getFirstIndexWithNonUniqueSolution() != 0) {
    throw error("expected first non-unique solution index to be 0");
  }
  if (graph.getLastIndexWithNonUniqueSolution() != graph.nOps() - 1) {
    throw error("expected last non-unique solution index to be nOps-1");
  }

  //    X      = unique solution index @0
  //  /   \
  // X     X
  //  \   /
  //    X      = unique solution index @3
  //    |
  //    X      = unique solution index @4
  //
  graph  = Graph();
  a      = graph.insertOp("a");
  b      = graph.insertOp("b");
  c      = graph.insertOp("c");
  auto d = graph.insertOp("d");
  auto e = graph.insertOp("e");

  graph.insertConstraint(a, b);
  graph.insertConstraint(a, c);
  graph.insertConstraint(b, d);
  graph.insertConstraint(c, d);
  graph.insertConstraint(d, e);
  graph.finalize();
  if (graph.getFirstIndexWithNonUniqueSolution() != 1 ||
      graph.getLastIndexWithNonUniqueSolution() != 2) {
    throw error("Misplaced non-unique solution indexs");
  }

  //
  //      a      : a unique-solution index, as only Op that can go at 0
  //    / | \
  //   b  |  c
  //   | /|  |
  //   e  |  d
  //    \ | /
  //      f---   : a unique-solution index, as only Op that can go at 5
  //     / \  |
  //    h   g |
  //     \ /  |
  //      i  /   : a unique-solution index, as only Op that can go at 8
  //      | /
  //      j      : a unique-solution index, as only Op that can go at 9
  //
  graph  = Graph();
  a      = graph.insertOp("a");
  b      = graph.insertOp("b");
  c      = graph.insertOp("c");
  d      = graph.insertOp("d");
  e      = graph.insertOp("e");
  auto f = graph.insertOp("f");
  auto g = graph.insertOp("g");
  auto h = graph.insertOp("h");
  auto i = graph.insertOp("i");
  auto j = graph.insertOp("j");

  graph.insertConstraint(a, b);
  graph.insertConstraint(a, c);
  graph.insertConstraint(a, f);
  graph.insertConstraint(b, e);
  graph.insertConstraint(c, d);
  graph.insertConstraint(d, f);
  graph.insertConstraint(e, f);
  graph.insertConstraint(f, g);
  graph.insertConstraint(f, h);
  graph.insertConstraint(f, j);
  graph.insertConstraint(g, i);
  graph.insertConstraint(h, i);
  graph.insertConstraint(i, j);

  graph.finalize();
  if (graph.getFirstIndexWithNonUniqueSolution() != 1 ||
      graph.getLastIndexWithNonUniqueSolution() != 7) {
    throw error("Misplaced non-unique solution indexs in more complex case");
  }
}
