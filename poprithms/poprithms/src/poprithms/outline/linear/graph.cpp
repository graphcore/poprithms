// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <sstream>
#include <tuple>
#include <unordered_set>

#include <poprithms/outline/linear/error.hpp>
#include <poprithms/outline/linear/graph.hpp>
#include <poprithms/outline/linear/logging.hpp>
#include <poprithms/schedule/supercon/graph.hpp>

namespace poprithms {
namespace outline {
namespace linear {

// forward edge of Graph, with raw uint64_t instead of OpIds
std::vector<std::vector<uint64_t>> Graph::getEdges_u64() const {
  std::vector<std::vector<uint64_t>> edges(nOps());
  for (const auto &op : allOps) {
    edges[op.id().get()].reserve(op.nOpsOut());
    for (auto opId : op.getOpsOut()) {
      edges[op.id().get()].push_back(opId.get());
    }
  }
  return edges;
}

// order couples of Graph, with raw uint64_t instead of OpIds
std::vector<std::array<uint64_t, 4>> Graph::getOrderCouples_u64() const {
  std::vector<std::array<uint64_t, 4>> orderCouples_u64;
  orderCouples_u64.reserve(orderCouples.size());
  for (const auto &x : orderCouples) {
    std::array<uint64_t, 4> x_u64{
        x[0].get(), x[1].get(), x[2].get(), x[3].get()};
    orderCouples_u64.push_back(x_u64);
  }
  return orderCouples_u64;
}

void Graph::setSchedule(SchedulingAlgorithm schedulingAlgorithm) {

  if (schedulingAlgorithm != SchedulingAlgorithm::Filo) {
    throw error("Only Filo scheduling algorithm is supported");
  }

  auto sched = schedule::supercon::getFiloSchedule(getEdges_u64(),
                                                   getOrderCouples_u64());

  schToOp.resize(sched.size());
  opToSch.resize(sched.size());

  for (uint64_t i = 0; i < sched.size(); ++i) {
    schToOp[i]        = OpId(sched[i]);
    opToSch[sched[i]] = i;
  }
}

Outline Graph::getOutline(
    const std::function<
        double(Type, const std::vector<std::tuple<Shape, DType>> &inTens)>
        & /*opCost*/,
    const std::function<double(uint64_t)> & /*copyCost*/,
    bool requireCommonExternalInputs,
    bool requireCommonExternalOutputs,
    OutliningAlgorithm outliningAlgorithm,
    SchedulingAlgorithm schedulingAlgorithm) {

  if (!requireCommonExternalInputs) {
    // TODO(T17667)
    throw error(
        "no support for !requireCommonExternalInputs yet, see T17667");
  }

  if (!requireCommonExternalOutputs) {
    // TODO(T17667)
    throw error(
        "no support for !requireCommonExternalOutputs yet, see T17667");
  }

  if (log().shouldLog(logging::Level::Debug)) {
    std::ostringstream oss;
    std::string lineStart = "\n       ";
    oss << "Graph::getOutline with " //
        << lineStart
        << "requireCommonExternalInputs  : " << requireCommonExternalInputs
        << lineStart
        << "requireCommonExternalOutputs : " << requireCommonExternalOutputs
        << lineStart <<                                                 //
        "OutliningAlgorithm  : " << outliningAlgorithm                  //
        << lineStart <<                                                 //
        "SchedulingAlgorithm : " << schedulingAlgorithm << lineStart << //
        "nOps in Graph       : " << nOps() << lineStart <<              //
        "nColors in Graph    : " << nColors() << lineStart <<           //
        "nTypes in Graph     : " << nTypes() << lineStart <<            //
        "nTensors in Graph   : " << nTensors();                         //
    log().debug(oss.str());
  }

  setSchedule(schedulingAlgorithm);

  if (outliningAlgorithm == OutliningAlgorithm::Algo2) {
    log().info(
        "Algo 2, proof of concept outliner is returning empty Outline");
    return Outline({}, nOps());
  } else {
    throw error(
        "Only OutliningAlgorithm::Algo2 is currently supported, see T19425");
  }
}

TensorId
Graph::insertTensor(const Shape &s, DType t, const std::string &dbs) {
  TensorId id{nTensors()};
  allTensors.push_back({s, t, id, dbs});
  return id;
}

OpId Graph::insertOp(Color c, Type t, const std::string &dbs) {
  OpId id{nOps()};
  allOps.push_back({c, id, t, dbs});
  return id;
}

void Graph::insertConstraint(OpId from, OpId to) {
  if (from >= nOps() || to >= nOps()) {
    std::ostringstream oss;
    oss << "Cannot insert constraint " << from << " -> " << to
        << ", as there are only " << nOps()
        << " Ops in this outline::linear::Graph";
    throw error(oss.str());
  }

  if (!get(from).hasOpOut(to)) {
    get(from).insertOpOut(to);
    get(to).insertOpIn(from);
  }
}

void Graph::insertOrderCouple(OpId a, OpId b, OpId c, OpId d) {
  orderCouples.push_back({a, b, c, d});
}

void Graph::insertIn(OpId opId, InIndex inIndex, TensorId tId) {
  get(opId).insertIn(tId, inIndex);
  if (!get(tId).hasOp(opId)) {
    get(tId).insertOp(opId);
  }
}

void Graph::insertOut(OpId opId, OutIndex outIndex, TensorId tId) {
  get(opId).insertOut(tId, outIndex);
  if (!get(tId).hasOp(opId)) {
    get(tId).insertOp(opId);
  }
}

void Graph::append(std::ostream &ost) const {
  ost << "Graph, with Ops:";
  for (const auto &op : allOps) {
    ost << "\n     " << op;
  }
  ost << "\nand with Tensors:";
  for (const auto &tensor : allTensors) {
    ost << "\n     " << tensor;
  }
}

std::ostream &operator<<(std::ostream &ost, const Graph &g) {
  g.append(ost);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, OutliningAlgorithm a) {
  switch (a) {
  case OutliningAlgorithm::Algo0: {
    ost << "OutliningAlgorithm::Algo0";
    return ost;
  }

  case OutliningAlgorithm::Algo1: {
    ost << "OutliningAlgorithm::Algo1";
    return ost;
  }

  case OutliningAlgorithm::Algo2: {
    ost << "OutliningAlgorithm::Algo2";
    return ost;
  }

  case OutliningAlgorithm::N:
  default:
    throw error("Invalid OutliningAlgorithm");
  }
}

std::ostream &operator<<(std::ostream &ost, SchedulingAlgorithm a) {
  switch (a) {
  case SchedulingAlgorithm::Filo: {
    ost << "SchedulingAlgorithm::Filo";
    return ost;
  }
  case SchedulingAlgorithm::N:
  default:
    throw error("Invalid SchedulingAlgorithm");
  }
}

uint64_t Graph::nTypes() const {
  std::unordered_set<Type> types;
  for (const auto &op : allOps) {
    types.insert(op.type());
  }
  return types.size();
}

uint64_t Graph::nColors() const {
  std::unordered_set<Color> colors;
  for (const auto &op : allOps) {
    colors.insert(op.color());
  }
  return colors.size();
}

} // namespace linear
} // namespace outline
} // namespace poprithms
