// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <testutil/memory/unwind/graph.hpp>

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace unwindtoy {

TensorId
Graph::input(const Shape &s, double linear, const std::string &name_) {
  TensorId out{createOp<Input>({}, {s}, linear), 0};
  setName(out, name_);
  return out;
}

Graph::~Graph() = default;

TensorId Graph::sum(const TensorIds &ins,
                    const std::vector<InIndex> &unwindIndices,
                    const memory::unwind::SumAttractions &sats) {
  return {createOp<Sum>(
              ins, {Shape::numpyVariadic(shapes(ins))}, unwindIndices, sats),
          0};
}

TensorId Graph::sum(const TensorIds &inIds,
                    const poprithms::memory::unwind::SumAttractions &satti) {
  const auto outShape = Shape::numpyVariadic(shapes(inIds));
  std::vector<InIndex> uws;
  for (uint64_t i = 0; i < inIds.size(); ++i) {
    if (shape(inIds[i]) == outShape) {
      uws.push_back(i);
    }
  }
  return sum(inIds, uws, satti);
}

Graph::Graph()
    : poprithms::common::schedulable::Graph(),
      singleGraph(createSubGraphId("oneGraph")) {}

Op::State Graph::getStartingState(const OpId opId,
                                  const TensorIds &inIds,
                                  const Shapes &outShapes) {
  const std::string name{};
  const std::vector<ConsumptionIds> consumptionIds(outShapes.size());
  MultioutOp::State st0(opId, inIds, consumptionIds, outShapes, name, *this);
  State st1(st0, singleGraph, {}, {});
  return st1;
}

OpId Graph::insertOp(std::unique_ptr<Op> createdOp) {
  const auto newId =
      poprithms::common::schedulable::Graph::insertSchedulableOp(
          std::move(createdOp));
  return newId;
}

OpId Graph::insertBinBoundary(poprithms::common::schedulable::SubGraphId) {
  throw poprithms::test::error(
      "Unimplemented method called. Context: insertBinBoundary");
}

void Graph::appendOpColumns(std::ostream &ost, const OpIds &opIds) const {

  auto cols  = getMultioutColumns(opIds);
  auto cols1 = getSchedulableColumns(opIds);
  cols.insert(cols.end(), cols1.cbegin(), cols1.cend());
  ost << alignedColumns(cols);
}

bool Graph::multiOutTypeSpecificEqualTo(
    const poprithms::common::multiout::Graph &) const {
  return true;
}

std::ostream &operator<<(std::ostream &ost, const Graph &g) {
  g.append(ost);
  return ost;
}

} // namespace unwindtoy
} // namespace poprithms
