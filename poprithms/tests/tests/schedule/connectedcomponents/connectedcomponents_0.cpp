// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/connectedcomponents/connectedcomponents.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms::schedule::connectedcomponents;

void assertConnectedComponents(
    const Edges<uint64_t> &edges,
    std::vector<std::vector<uint64_t>> expectedPartition) {

  ConnectedComponents cc(edges);

  for (const auto &ep : expectedPartition) {
    if (ep.empty()) {
      throw poprithms::test::error("Empty partition: not expected");
    }

    // Each vector in expectedPartitions contains a group of nodes (by global
    // id) which are expected to be in the same partition. We check that the
    // elements in such a vector are also in the same partition in the
    // constructed ConnectedComponents. Note that we are not checking for
    // equivalent ComponentIds between the expected and observed partitions,
    // rather that the groupings are the same.
    for (auto x : ep) {
      if (cc.componentId(x) != cc.componentId(ep[0])) {
        throw poprithms::test::error("Components do not agree");
      }
    }
  }

  // We confirm that the number of partitions is the same.
  if (expectedPartition.size() != cc.nComponents()) {
    throw poprithms::test::error("number of components do not agree");
  }
}

void test0() {

  // 0->1->2->3
  assertConnectedComponents({{1}, {2}, {3}, {}}, {{0, 1, 2, 3}});

  // isolated nodes
  assertConnectedComponents({{}, {}, {}}, {{0}, {1}, {2}});

  // Test of non-DAG
  // 0 <-> 1   2 <-> 3
  assertConnectedComponents({{1}, {0}, {3}, {2}}, {{0, 1}, {2, 3}});

  // 0 <- 1 <- 2 -> 3 -> 4
  assertConnectedComponents({{}, {0}, {1, 3}, {4}, {}}, {{0, 1, 2, 3, 4}});

  // 0 -> 1 <- 2 -> 6
  // 3 <-> 4 -> 5
  assertConnectedComponents({{1}, {}, {1, 6}, {4}, {3, 5}, {}, {}},
                            {{0, 1, 2, 6}, {3, 4, 5}});
}

void test1() {
  bool caught = false;
  try {
    ConnectedComponents cc(Edges<int64_t>({{1}, {-1}}));
  } catch (const poprithms::error::error &e) {
    caught = true;
  }

  if (!caught) {
    throw poprithms::test::error("Failed to catch negative edge");
  }
}

void test2() {
  bool caught = false;
  try {
    ConnectedComponents cc(Edges<int64_t>({{1}, {100}}));
  } catch (const poprithms::error::error &e) {
    caught = true;
  }

  if (!caught) {
    throw poprithms::test::error("Failed to catch too-large edge end");
  }
}

void test3() {

  // Prints:
  // In component 0 : (0,1,2,3)
  // In component 1 : (4,5)

  Edges<int64_t> a{{1}, {2}, {3}, {0}, {5}, {}};
  std::cout << ConnectedComponents(a) << std::endl;
}

} // namespace

int main() {

  test0();
  test1();
  test2();
  test3();

  return 0;
}
