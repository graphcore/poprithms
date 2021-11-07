// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <poprithms/error/error.hpp>
#include <poprithms/util/copybyclone_impl.hpp>

namespace test {

using namespace poprithms;

struct Node {
  Node(int id) : id_(id) {}
  int id() const { return id_; }
  void setId(int i) { id_ = i; }
  bool operator==(const Node &rhs) const { return id() == rhs.id(); }
  std::unique_ptr<Node> clone() const {
    return std::make_unique<Node>(*this);
  }

private:
  int id_;
};

using UpNode = util::CopyByClone<Node>;

struct DerivedNode : public Node {
  DerivedNode() : Node(5) {}
};

struct Graph {
  Graph() = default;
  int gid;
  std::string gname;
  std::vector<UpNode> nodes;
  bool operator==(const Graph &rhs) const { return nodes == rhs.nodes; }

  void insert(int id) { nodes.push_back(UpNode(std::make_unique<Node>(id))); }

  void insertNull() { nodes.push_back(UpNode(nullptr)); }

  // We can implicitly up-cast from DerivedNode to Node. Note that making the
  // CopyByClone constructor explicit would prevent this.
  void insertDerivedNode() {
    nodes.push_back(UpNode(std::make_unique<DerivedNode>()));
  }
};

} // namespace test

void test0() {
  using namespace test;
  auto a1 = UpNode(std::make_unique<Node>(1));
  auto a2 = UpNode(std::make_unique<Node>(2));
  auto a3 = UpNode(std::make_unique<Node>(3));
  auto a4 = UpNode(std::make_unique<Node>(4));

  // Assignment operators
  a3 = a1;
  if (a3.uptr->id() != 1) {
    throw poprithms::test::error("incorrect value from copy assignment");
  }
  a4 = std::move(a2);
  if (a4.uptr->id() != 2) {
    throw poprithms::test::error("incorrect value from move assignment");
  }

  // Constructors
  auto a5 = a3;
  if (a5.uptr->id() != 1) {
    throw poprithms::test::error("incorrect value from copy constructor");
  }
  auto a6 = std::move(a4);
  if (a6.uptr->id() != 2) {
    throw poprithms::test::error("incorrect value from move constructor");
  }

  Graph g;
  g.insert(1);
  g.insert(2);
  g.insert(3);
  g.insertDerivedNode();

  auto g2 = g;
  if (!(g2 == g)) {
    throw poprithms::test::error(
        "Directly after copying, graphs should compare equal");
  }
  g2.nodes[0].uptr->setId(100);
  if (g2 == g) {
    throw poprithms::test::error(
        "After g2 has been modified, the graphs should not compare equal. "
        "Note that copying Graph g involved cloning all of its Nodes, so "
        "when a Node in the copy was modified, this had no effect on g. If a "
        "user wants different behaviour, where a resource is shared across "
        "Graphs, they should use std::shared_ptr. ");
  }
}

void testNullPtr0() {
  // nullptr:
  auto a = test::UpNode();

  // copy constructor
  auto b = a;
  auto c = a;

  // move constructor
  auto d = std::move(b);
  auto e = std::move(c);

  // assignment operator
  d = e;

  // move operator
  d = std::move(a);

  if (d.uptr) {
    throw poprithms::test::error("Expected nullptr to be retained");
  }
}

void testNullPtr1() {
  test::Graph g;
  g.insert(1);
  g.insertDerivedNode();
  g.insertNull();

  auto g2 = g;
  auto g3 = g;
  g3      = g2;
  auto g4 = std::move(g3);

  if (!(g4 == g)) {
    throw poprithms::test::error(
        "g4 is a copy of g, and should compare equal");
  }
}

int main() {
  test0();
  testNullPtr0();
  testNullPtr1();
  return 0;
}
