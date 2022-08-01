// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <set>
#include <vector>

#include <poprithms/coloring/ipropagator.hpp>

namespace {

// Interface completion with Node=uint32_t and Color=std::string
class StringTestPropagator final
    : public poprithms::coloring::IPropagator<uint32_t, std::string> {

public:
  StringTestPropagator() = default;

  void addEdge(uint32_t a, uint32_t b) {
    insert(fwd, a, b);
    insert(bwd, b, a);
  }

  std::string nodeString(uint32_t n) const {
    return "node-" + std::to_string(n);
  }

  std::vector<uint32_t> ins(uint32_t a) const final { return get(bwd, a); }
  std::vector<uint32_t> outs(uint32_t a) const final { return get(fwd, a); }

private:
  using Map = std::map<uint32_t, std::set<uint32_t>>;
  Map fwd;
  Map bwd;

  void insert(Map &m, uint32_t a, uint32_t b) {
    auto found = m.find(a);
    if (found != m.cend()) {
      found->second.insert(b);
    } else {
      m.insert({a, {b}});
    }
  }

  std::vector<uint32_t> get(const Map &m, uint32_t a) const {
    auto found = m.find(a);
    if (found == m.cend()) {
      return {};
    }
    auto &&x = found->second;
    return {x.cbegin(), x.cend()};
  }
};

StringTestPropagator getBiDiamond() {

  //
  //      +-- 1 --+
  //  0 --+       +-- 3
  //      +-- 2 --+
  //  4 --+       +-- 6
  //      +-- 5 --+
  //

  StringTestPropagator f;
  f.addEdge(0, 1);
  f.addEdge(0, 2);

  f.addEdge(1, 3);

  f.addEdge(2, 3);
  f.addEdge(2, 6);

  f.addEdge(4, 2);
  f.addEdge(4, 5);

  f.addEdge(5, 6);

  return f;
}

void test0() {
  auto f = getBiDiamond();
  f.setAndPropagateForward(1, "one");
  f.setAndPropagateForward(0, "zero");
  if (!(f.color(3) == "one" && f.color(1) == "one" && f.color(2) == "zero" &&
        f.color(0) == "zero")) {
    throw poprithms::test::error(
        "Expected diamond to be split into 2 groups of 2 elements.");
  }
}

void test1() {
  auto f = getBiDiamond();
  f.setColor(3, "c");
  f.propagateForwardAndBackward(3);
  for (auto x : {0, 1, 2, 3, 4, 5, 6}) {
    if (f.color(x) != "c") {
      throw poprithms::test::error(
          "Expected all to be nodes to have color 'c'");
    }
  }

  if (f.allWithColor("c").size() != 7) {
    throw poprithms::test::error("Expected all 7 nodes to have color 'c'.");
  }
}

void test2() {
  auto f = getBiDiamond();
  f.setColor(2, "2");
  f.propagateForward(2);
  f.propagateBackward(2);
  f.setColor(1, "1");
  f.propagateForwardAndBackward(1);
  f.setColor(5, "5");
  if (f.allWithColor("2").size() != 5 || f.allWithColor("5").size() != 1 ||
      f.allWithColor("1").size() != 1) {
    throw poprithms::test::error("Expected 5 nodes to have color '2', and "
                                 "each of the other colors to have 1 node.");
  }
}

void test3() {
  auto f = getBiDiamond();
  f.setColor(1, "x");
  f.setColor(5, "x");
  f.flushForward("x", "y", [](auto) { return true; });
  if (f.color(3) != "y" || f.color(6) != "y") {
    throw poprithms::test::error(
        "Expected nodes 3 and 6 to be on the flush path (i.e. to have colors "
        "3 and 6 respectively).");
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
