// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <poprithms/common/multiout/error.hpp>
#include <poprithms/common/multiout/graph.hpp>

namespace {
using namespace poprithms::common;

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
  multiout::OpId grow() {
    const multiout::Op::State s(nOps(), {}, {}, {}, {}, "");
    return insertMultioutOp(std::make_unique<test::Op>(s));
  }
  virtual ~Graph() override = default;

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
    collected.push_back(g.grow());
  }
  if (collected[34] != multiout::OpId(34)) {
    throw multiout::error("Expected OpIds to increment by 1, starting at 0");
  }
}

} // namespace

int main() {
  test0();
  return 0;
}
