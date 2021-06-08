// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_set>

#include <poprithms/common/multiout/consumptionid.hpp>
#include <poprithms/common/multiout/error.hpp>
#include <poprithms/common/multiout/graph.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/stringutil.hpp>

namespace {
using namespace poprithms::common;
using Shape = poprithms::ndarray::Shape;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using Shapes = poprithms::ndarray::Shapes;

namespace test {

class Op : public multiout::Op {
public:
  Op(const multiout::Op::State &s) : multiout::Op(s) {}
  std::string typeString() const final { return "LazyMauveOp"; }
  std::unique_ptr<multiout::Op> clone() const final {
    return std::make_unique<Op>(*this);
  }

private:
  bool multiOutTypeSpecificEqualTo(const multiout::Op &) const final {
    return true;
  };
};

class Graph : public multiout::Graph {
public:
  multiout::OpId insert(const TensorIds &inIds, uint64_t nOuts) {
    const Shapes outShapes(nOuts, Shape({1}));
    std::vector<multiout::ConsumptionIds> consOut(nOuts);
    const multiout::Op::State s(
        nOps(), inIds, consOut, shapes(inIds), outShapes, "");
    return insertMultioutOp(std::make_unique<test::Op>(s));
  }
  virtual ~Graph() override = default;
  void append(std::ostream &) const;

private:
  bool multiOutTypeSpecificEqualTo(const multiout::Graph &) const final {
    return true;
  }
};

} // namespace test

void test0() {
  test::Graph g;
  g.setName("my_test_graph");
  if (g.getName() != "my_test_graph") {
    throw multiout::error(
        "Failed to correctly set and get name in test::Graph class. ");
  }

  multiout::OpIds collected;
  for (uint64_t i = 0; i < 50; ++i) {
    collected.push_back(g.insert({}, 0));
  }
  if (collected[34] != multiout::OpId(34)) {
    throw multiout::error("Expected OpIds to increment by 1, starting at 0");
  }
}

void test::Graph::append(std::ostream &ost) const {
  const auto c = getMultioutColumns();
  ost << alignedColumns(c);
}

std::ostream &operator<<(std::ostream &ost, const test::Graph &g) {
  g.append(ost);
  return ost;
}

void testLogging0() {

  //  OpId OpType      InTensors OutIndex Shape
  //  ---- ------      --------- -------- -----
  //  0    LazyMauveOp ()
  //  1    LazyMauveOp ()        0        (1)
  //                             1        (1)
  //                             2        (1)
  //  2    LazyMauveOp ()        0        (1)
  //  3    LazyMauveOp ()

  test::Graph g;
  g.insert({}, 0);
  g.insert({}, 3);
  g.insert({}, 1);
  g.insert({}, 0);
  std::cout << g << std::endl;

  const auto outCols = g.getMultioutColumns();
  if (outCols.empty()) {
    throw multiout::error("No multiout columns in test");
  }
  for (auto x : outCols) {
    if (x.nEntries() != outCols[0].nEntries()) {
      throw multiout::error("size of each column should be the same");
    }
  }

  // expect 2 blanks in Shape column:
  for (auto col : outCols) {
    auto vs = col.entries();
    if (std::any_of(
            vs.cbegin(), vs.cend(), [](auto s) { return s == "Shape"; })) {

      auto isSpace = [](std::string s) {
        return std::all_of(
            s.cbegin(), s.cend(), [](auto x) { return std::isspace(x); });
      };

      int cnt = std::accumulate(
          vs.cbegin(),
          vs.cend(),
          int(0),
          [&isSpace](int a, const std::string &b) { return a + isSpace(b); });

      if (cnt != 2) {
        throw multiout::error("Expected 2 empty rows in the Shape column");
      }
    }
  }
}

void testInsAndOuts() {

  test::Graph g;
  auto a = g.insert({}, 2);
  auto b = g.insert({}, 3);
  auto c = g.insert({{a, 0}, {b, 1}, {b, 2}}, 4);

  auto insNouts = g.inAndOutTensorIds(c);

  std::sort(insNouts.begin(), insNouts.end());

  if (insNouts !=
      TensorIds({{a, 0}, {b, 1}, {b, 2}, {c, 0}, {c, 1}, {c, 2}, {c, 3}})) {
    throw multiout::error("Incorrect input+output TensorIds");
  }
}

void testHashTensorId() {
  std::unordered_set<uint64_t> nDistintHashes;
  uint64_t nTensors{10000};

  std::mt19937_64 gen(1011);
  for (uint64_t i = 0; i < nTensors; ++i) {

    TensorId tId;
    auto k = gen() % 3;

    // random OpId and OutIndex
    if (k == 0) {
      tId = TensorId(static_cast<uint32_t>(gen()),
                     static_cast<uint32_t>(gen()));
    }

    // repeated OpIds, random OutIndex
    else if (k == 1) {
      tId = TensorId(gen() % 3, static_cast<uint32_t>(gen()));
    }

    // repeates OutIndex, random OpIds
    else {
      tId = TensorId(static_cast<uint32_t>(gen()), gen() % 3);
    }
    nDistintHashes.emplace(std::hash<TensorId>{}(tId));
  }

  std::cout << nDistintHashes.size() << std::endl;

  if (static_cast<double>(nDistintHashes.size()) /
          static_cast<double>(nTensors) <
      0.99) {
    throw poprithms::common::multiout::error("Failed hash test");
  }
}

} // namespace

int main() {
  test0();
  testLogging0();
  testInsAndOuts();
  testHashTensorId();
  return 0;
}
