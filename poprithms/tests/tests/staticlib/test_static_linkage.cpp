#include <poprithms/schedule/anneal/graph.hpp>
#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>

int main() {
  poprithms::schedule::anneal::Graph g;
  g.insertAlloc(31415.);
  poprithms::schedule::transitiveclosure::TransitiveClosure g2({});
  return 0;
}
