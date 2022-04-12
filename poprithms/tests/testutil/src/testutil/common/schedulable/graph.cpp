// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <testutil/common/schedulable/graph.hpp>

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace common {
namespace schedulable_test {

Op::Op(const schedulable::Op::State &s, bool phobic)
    : phobic_(phobic), schedulable::Op(s) {}

std::string Op::typeString() const { return "schedulable_test::Op"; }

std::unique_ptr<multiout::Op> Op::cloneMultioutOp() const {
  return std::make_unique<Op>(*this);
}

OpId Graph::insert(const TensorIds &ins,
                   uint64_t nOut,
                   SubGraphId sgId,
                   const std::string &name,
                   bool isPhobic) {

  const Shapes outShapes(nOut, Shape({}));
  const Shapes inShapes = shapes(ins);
  std::vector<multiout::ConsumptionIds> outCons(nOut);

  const multiout::Op::State baseState(
      OpId(nxtOpId()), ins, outCons, outShapes, name, *this);

  OpIds inNonDataDeps{};
  OpIds outNonDataDeps{};

  const schedulable::Op::State state(
      baseState, sgId, inNonDataDeps, outNonDataDeps);

  auto opId = insertSchedulableOp(std::make_unique<Op>(state, isPhobic));

  return opId;
}

Graph::~Graph() = default;

void Graph::appendOpColumns(std::ostream &ost, const OpIds &opIds) const {
  auto cols = getMultioutColumns(opIds, {});
  for (auto c : getSchedulableColumns(opIds, {})) {
    cols.push_back(c);
  }
  ost << alignedColumns(cols);
}

OpId Graph::insertBinBoundary(schedulable::SubGraphId sgId) {
  return insert({}, 0, sgId, "binBoundary");
}

std::ostream &operator<<(std::ostream &ost, const Graph &g) {
  g.append(ost);
  return ost;
}

} // namespace schedulable_test
} // namespace common
} // namespace poprithms
