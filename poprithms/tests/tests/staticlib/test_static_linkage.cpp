#include <poprithms/schedule/anneal/graph.hpp>
#include <poprithms/schedule/pathmatrix/pathmatrix.hpp>

int main() {
  poprithms::schedule::anneal::Graph g;
  g.insertAlloc(31415.);
  poprithms::schedule::pathmatrix::PathMatrix g2({});
  return 0;
}
