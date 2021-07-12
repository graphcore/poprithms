// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/alias/graph.hpp>

int main() {

  using namespace poprithms::memory::alias;
  using namespace poprithms::memory::nest;

  Graph g;

  const auto foo = g.tensor(g.allocate({5, 10, 15}))
                       .broadcast(6, 0)
                       .broadcast(3, 1)
                       .broadcast(2, 2);
  if (foo.shape() != Shape{30, 30, 30} || !foo.containsAliases()) {
    throw poprithms::test::error("Failure in basic broadcasting test: foo");
  }

  const auto bar0 = foo.slice({0, 0, 0}, {5, 10, 15});
  if (bar0.shape() != Shape{5, 10, 15} || bar0.containsAliases()) {
    throw poprithms::test::error("Failure in basic broadcasting test: bar0");
  }

  const auto bar1 = foo.slice({1, 1, 1}, {6, 11, 16});
  if (bar1.shape() != Shape{5, 10, 15} || bar1.containsAliases()) {
    std::cout << g.verboseString() << std::endl;
    throw poprithms::test::error("Failure in basic broadcasting test: bar1");
  }

  const auto bar2 = foo.slice({1, 1, 1}, {11, 2, 2});
  if (bar2.shape() != Shape{10, 1, 1} || !bar2.containsAliases()) {
    std::cout << g.verboseString() << std::endl;
    throw poprithms::test::error("Failure in basic broadcasting test: bar2");
  }

  const auto bar3 = foo.slice({7, 1, 1}, {13, 2, 2});
  if (bar3.shape() != Shape{6, 1, 1} || !bar3.containsAliases()) {
    std::cout << g.verboseString() << std::endl;
    throw poprithms::test::error("Failure in basic broadcasting test: bar3");
  }

  return 0;
}
