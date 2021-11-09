// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <testutil/common/schedulable/schedulablegraph.hpp>

namespace poprithms {
namespace common {
namespace schedulable_test {

Op::Op(const schedulable::Op::State &s) : schedulable::Op(s) {}
std::string Op::typeString() const { return "ScazooOp"; }
std::unique_ptr<multiout::Op> Op::cloneMultioutOp() const {
  return std::make_unique<Op>(*this);
}

bool Op::schedulableTypeSpecificEqualTo(const schedulable::Op &) const {
  return true;
}

OpId Graph::insert(const TensorIds &ins,
                   uint64_t nOut,
                   SubGraphId sgId,
                   std::string name) {

  const Shapes outShapes(nOut, Shape({}));
  const Shapes inShapes = shapes(ins);
  std::vector<multiout::ConsumptionIds> outCons(nOut);

  const multiout::Op::State baseState(
      OpId(nxtOpId()), ins, outCons, outShapes, name, *this);

  OpIds inNonDataDeps{};
  OpIds outNonDataDeps{};

  const schedulable::Op::State state(
      baseState, sgId, inNonDataDeps, outNonDataDeps);

  auto opId = insertSchedulableOp(std::make_unique<Op>(state));

  return opId;
}

Graph::~Graph() = default;
void Graph::appendOpColumns(std::ostream &ost, const OpIds &opIds) const {
  auto cols = getMultioutColumns(opIds);
  for (auto c : getSchedulableColumns(opIds)) {
    cols.push_back(c);
  }
  ost << alignedColumns(cols);
}

OpId Graph::insertBinBoundary(schedulable::SubGraphId sgId) {
  return insert({}, 0, sgId, "binBoundary");
}

bool Graph::multiOutTypeSpecificEqualTo(const multiout::Graph &) const {
  return true;
}

std::ostream &operator<<(std::ostream &ost, const Graph &g) {
  g.append(ost);
  return ost;
}

} // namespace schedulable_test
} // namespace common
} // namespace poprithms
