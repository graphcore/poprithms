// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/logging/logging.hpp>
#include <poprithms/outline/linear/graph.hpp>

int main() {

  using namespace poprithms::outline::linear;

  // TODO(T19597) should be implemented so we don't need to do this.
  poprithms::logging::setGlobalLevel(poprithms::logging::Level::Debug);

  Graph graph;

  auto op0 = graph.insertOp(Color{0}, Type{0}, "op0");
  auto op1 = graph.insertOp(Color{1}, Type{1}, "op1");
  auto op2 = graph.insertOp(Color{1}, Type{0}, "op2");

  graph.insertConstraint(op0, op1);
  graph.insertConstraint(op1, op2);

  auto t0 = graph.insertTensor({2, 3}, DType::INT32, "t0");
  auto t1 = graph.insertTensor({1, 3}, DType::INT32, "t1");
  auto t2 = graph.insertTensor({2, 1}, DType::INT32, "t2");

  graph.insertOut(op0, 0, t0);
  graph.insertOut(op1, 0, t1);
  bool didCatch = false;
  try {
    graph.insertOut(op1, 0, t2);
  } catch (const poprithms::error::error &e) {
    didCatch = true;
  }
  if (!didCatch) {
    throw poprithms::test::error(
        "Failed to catch case of duplicate Tensor inserts index");
  }

  graph.insertOut(op1, 2, t2);
  graph.insertIn(op1, 0, t0);
  graph.insertIn(op2, 3, t1);
  graph.insertIn(op2, 2, t2);

  auto outline = graph.getOutline(
      [](Type, const std::vector<std::tuple<Shape, DType>> &) { return 1.; },
      [](uint64_t) { return 0.; },
      true,
      true,
      OutliningAlgorithm::Algo2,
      SchedulingAlgorithm::Filo);

  std::cout << graph << std::endl;

  return 0;
}
