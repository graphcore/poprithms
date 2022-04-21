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

  cols.push_back({"Copy ins", std::move(copySources__), {}});
  cols.push_back({"Copy outs", std::move(copyDestinations__), {}});

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

  auto oldCopyIns = op(opId).inCopies().copyIns();
  std::vector<CopyIn> copyIns;
  for (InIndex i = 0; i < op(opId).inCopies().nInTensors(); ++i) {
    if (!coin.isRemoved(i)) {
      copyIns.push_back(oldCopyIns.at(i.get()));
    }
  }
  mutableOp(opId).inCopies_ = CopyIns(copyIns);
}

bool Graph::isCarriedTo(const TensorId &tId, const CallStack &cs) const {
  if (cs.empty()) {
    return false;
  }
  return op(cs.back().caller()).isCarriedTo(tId);
}

TensorId Graph::carriedFrom(const TensorId &tId, const CallStack &cs) const {

  if (cs.empty()) {
    std::ostringstream oss;
    oss << "Invalid call to carriedFrom with tId=" << tId
        << ": call stack empty.";
    throw poprithms::test::error(oss.str());
  }
  return op(cs.back().caller()).carriedFrom(tId);
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

  std::vector<std::pair<TensorId, TensorId>> carries{};

  auto op_ = std::make_unique<Op>(getState(ins, nOut, sgId, name),
                                  SubGraphIds{},
                                  CopyIns{},
                                  CopyOuts(std::vector<TensorIds>(nOut)),
                                  carries);

  return insertSchedulableOp(std::move(op_));
}

std::ostream &operator<<(std::ostream &ost, const Graph &g) {
  g.append(ost);
  return ost;
}

std::unique_ptr<multiout::Op> Op::cloneMultioutOp() const {
  return std::make_unique<Op>(
      getSchedulableState(), callees(), inCopies(), outCopies(), carries_);
}

OpId Graph::insert(SubGraphId sgId,
                   const SubGraphIds &callees,
                   const CopyIns &inCopies,
                   const CopyOuts &outCopies,
                   OptionalTensorId condition,
                   const std::vector<std::pair<TensorId, TensorId>> &carries,
                   const std::string &name) {

  if ((outCopies.nOutTensors() != 0) &&
      (callees.size() != outCopies.nCallees())) {
    std::ostringstream oss;
    oss << "Callees is of size " << callees.size()
        << " but outCopies reports " << outCopies.nCallees() << ".";
    throw poprithms::test::error(oss.str());
  }

  if (callees.empty()) {
    throw poprithms::test::error(
        "No callees: use the other insert method of this mock class.");
  }

  TensorIds inIds = inCopies.srcIds();

  if (condition.has_value()) {
    inIds.push_back(condition.value());
  }

  const auto nOuts = outCopies.nOutTensors();
  const auto state = getState(inIds, nOuts, sgId, name);

  return insertSchedulableOp(
      std::make_unique<Op>(state, callees, inCopies, outCopies, carries));
}

InIndices Op::nonCalleeCopyInIndices() const {
  // all ops with calles are switch-like ops with the final input 0 being
  // the condition tensor.
  if (!callees_.empty()) {
    if (inCopies_.nInTensors() + 1 == nInTensors()) {
      return {nInTensors() - 1};
    } else if (inCopies_.nInTensors() == nInTensors()) {
      return {};
    } else {
      throw poprithms::test::error("Mock class logic error: can only have 1 "
                                   "non-copy tensor with callees.");
    }
  }
  return inIndices();
}

bool Op::isCarriedTo(const TensorId &tId) const {
  for (auto p : carries_) {
    if (p.second == tId) {
      return true;
    }
  }
  return false;
}

TensorId Op::carriedFrom(const TensorId &to) const {
  for (auto p : carries_) {
    if (p.second == to) {
      return p.first;
    }
  }
  std::ostringstream oss;
  oss << "The tensor " << to << " is not carried to.";
  throw poprithms::test::error(oss.str());
}

std::vector<std::pair<InIndex, TensorId>> Op::copyInDsts() const {
  if (callees_.empty()) {
    return {};
  }

  std::vector<std::pair<InIndex, TensorId>> ps;
  for (uint64_t i = 0; i < inCopies().nInTensors(); ++i) {
    ps.push_back({i, inCopies().dst(i)});
  }
  return ps;
}

} // namespace callstack_test
} // namespace program
} // namespace poprithms
