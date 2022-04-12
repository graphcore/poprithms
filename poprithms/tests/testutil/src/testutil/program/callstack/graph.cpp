// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <memory>
#include <sstream>

#include <testutil/program/callstack/graph.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/program/callstack/copymap.hpp>

namespace poprithms {
namespace program {
namespace callstack_test {

using namespace poprithms::program::callstack_test;

Graph::~Graph() = default;

OpId Graph::insertBinBoundary(schedulable::SubGraphId sgId) {
  return insert({}, 0, sgId, "binBoundary");
}

void Graph::appendOpColumns(std::ostream &ost, const OpIds &opIds_) const {

  auto cols = getMultioutColumns(opIds_, {});
  for (auto c : getSchedulableColumns(opIds_, {})) {
    cols.push_back(c);
  }

  const auto nTens = nMultioutRows(opIds_);
  using Strings    = std::vector<std::string>;
  // extensions:
  Strings copySources__(nTens, "");
  Strings copyDestinations__(nTens, "");

  uint64_t ti{0};
  for (auto i : opIds_) {
    const auto subGraphId = op(i).subGraphId();
    const auto gName      = subGraphName(subGraphId);
    copySources__[ti]     = op(i).inCopies().str();
    for (uint64_t o = 0; o < op(i).nOutTensors(); ++o) {
      copyDestinations__[ti] = op(i).outCopies().outSourcesString(o);
      ++ti;
    }
    if (op(i).nOutTensors() == 0) {
      ++ti;
    }
  }

  cols.push_back({"Copy Sources", std::move(copySources__), {}});
  cols.push_back({"Copy Destinations", std::move(copyDestinations__), {}});

  ost << alignedColumns(cols);
}

std::string Op::typeString() const {
  std::ostringstream oss;
  oss << "callstack_test::Op";
  util::append(oss, callees_);
  return oss.str();
}

void Graph::multiOutTypeSpecificRemoveOutputs(
    OpId opId,
    const ContiguousOutIndexSubset &coin,
    const OptionalTensorIds &) {
  mutableOp(opId).outCopies_.reduce(coin);
}

void Graph::multiOutTypeSpecificRemoveInputs(
    OpId opId,
    const ContiguousInIndexSubset &coin) {
  auto copies = mutableOp(opId).inCopies_.copyIns();
  coin.reduce(copies);
  mutableOp(opId).inCopies_ = CopyIns(copies);
}

schedulable::Op::State Graph::getState(const TensorIds &ins,
                                       uint64_t nOut,
                                       const SubGraphId sgId,
                                       const std::string &name) const {

  const Shapes outShapes(nOut, Shape({}));
  const Shapes inShapes = shapes(ins);
  std::vector<multiout::ConsumptionIds> outCons(nOut);

  const multiout::Op::State baseState(
      OpId(nxtOpId()), ins, outCons, outShapes, name, *this);

  OpIds inNonDataDeps{};
  OpIds outNonDataDeps{};

  const schedulable::Op::State state(
      baseState, sgId, inNonDataDeps, outNonDataDeps);
  return state;
}

OpId Graph::insert(const TensorIds &ins,
                   uint64_t nOut,
                   SubGraphId sgId,
                   const std::string &name) {

  return insertSchedulableOp(
      std::make_unique<Op>(getState(ins, nOut, sgId, name),
                           SubGraphIds{},
                           CopyIns{},
                           CopyOuts(std::vector<TensorIds>(nOut))));
}

std::ostream &operator<<(std::ostream &ost, const Graph &g) {
  g.append(ost);
  return ost;
}

std::unique_ptr<multiout::Op> Op::cloneMultioutOp() const {
  return std::make_unique<Op>(
      getSchedulableState(), callees(), inCopies(), outCopies());
}

OpId Graph::insert(SubGraphId sgId,
                   const SubGraphIds &callees,
                   const CopyIns &inCopies,
                   const CopyOuts &outCopies,
                   const std::string &name) {

  if ((outCopies.nOutTensors() != 0) &&
      (callees.size() != outCopies.nCallees())) {
    throw poprithms::test::error("Callees and out copies must be same size");
  }

  if (callees.empty()) {
    throw poprithms::test::error("No callees, use other insert method");
  }

  const auto nOuts = outCopies.nOutTensors();
  const auto state = getState(inCopies.srcIds(), nOuts, sgId, name);

  return insertSchedulableOp(
      std::make_unique<Op>(state, callees, inCopies, outCopies));
}

} // namespace callstack_test
} // namespace program
} // namespace poprithms
