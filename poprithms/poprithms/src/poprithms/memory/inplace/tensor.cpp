// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "ops.hpp"

#include <sstream>

#include <memory/inplace/error.hpp>

#include <poprithms/common/multiout/util.hpp>
#include <poprithms/memory/alias/graph.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

Tensor Tensor::withName(const std::string &dbs) const {
  setName(dbs);
  return *this;
}

TensorIds Tensor::tensorIds(const Tensors &tensors) {
  return common::multiout::util::ids<Tensors, TensorIds>(tensors);
}

OpIds Tensor::opIds(const Tensors &tensors) {
  OpIds opIds(tensors.size());
  for (uint64_t i = 0; i < tensors.size(); ++i) {
    opIds[i] = tensors[i].opId();
  }
  return opIds;
}

Tensor Tensor::concat(const Tensors &ts, uint64_t axis) {
  if (ts.empty()) {
    throw error("Cannot concatenate empty vector of Tensors. ");
  }
  if (std::any_of(ts.cbegin(), ts.cend(), [&ts](const auto &t) {
        return &t.graph() != &ts[0].graph();
      })) {
    throw error("Cannot concatenate Tensors from different Graphs");
  }
  return {ts[0].graph().concat(tensorIds(ts), axis), ts[0].graph()};
}

void Tensor::assertSameGraph(const Tensors &ts) {
  if (ts.empty()) {
    return;
  }
  if (std::any_of(ts.cbegin(), ts.cend(), [&ts](const auto &t) {
        return t.graph_ != ts[0].graph_;
      })) {
    std::ostringstream oss;
    oss << "Failed in Tensor::assertSameGraph where Tensors are " << ts;
    throw error(oss.str());
  }
}

Tensor Tensor::aliasGate(const Tensors &ts, InIndex inIndex) {
  if (ts.empty()) {
    throw error("Cannot create AliasGate with 0 inputs");
  }
  assertSameGraph(ts);
  return {ts[0].graph().aliasGate(tensorIds(ts), inIndex), ts[0].graph()};
}

Tensor Tensor::aliasGate(const Tensors &ts) {
  if (ts.empty()) {
    throw error("Cannot create AliasGate with 0 inputs");
  }
  return {ts[0].graph().aliasGate(tensorIds(ts)), ts[0].graph()};
}

Tensors Tensor::multi(Graph &g,
                      const Tensors &ins,
                      const Shapes &outs,
                      const CrossLinks &cas) {

  if (std::any_of(ins.cbegin(), ins.cend(), [&g](const auto &t) {
        return &t.graph() != &g;
      })) {
    throw error("Ins to multi must have same graph as input Graph");
  }

  const auto opId = g.multi(tensorIds(ins), outs, cas);
  Tensors outTensors;
  outTensors.reserve(outs.size());
  for (const auto outId : g.op(opId).outTensorIds()) {
    outTensors.push_back({outId, g});
  }
  return outTensors;
}

Tensor Tensor::slice(const Lower &l, const Upper &u) const {
  return {graph().slice(id(), l, u), graph()};
}

Tensor Tensor::settSample(const Region &r) const {
  return {graph().settSample(id(), r), graph()};
}

Tensor Tensor::aliasGate(bool isOpen) const {
  if (!isOpen) {
    return {graph().aliasGate({id()}), graph()};
  }
  return {graph().aliasGate({id()}, 0), graph()};
}

Tensor Tensor::modify() const { return {graph().modify(id()), graph()}; }

Tensor Tensor::reshape(const Shape &outShape) const {
  return {graph().reshape(id(), outShape), graph()};
}

Tensor Tensor::flatten() const { return reshape(shape().flatten()); }

Tensor Tensor::subSample(Stride stride, Dimension dimension) const {
  return {graph().subSample(id(), stride, dimension), graph()};
}

Tensor Tensor::subSample(const Strides &strides) const {
  return {graph().subSample(id(), strides), graph()};
}

Tensor Tensor::dimShuffle(const Permutation &perm) const {
  return {graph().dimShuffle(id(), perm), graph()};
}

Tensor Tensor::reverse(const Dimensions &dims) const {
  return {graph().reverse(id(), dims), graph()};
}

Tensor Tensor::expand(const Shape &outShape) const {
  return {graph().expand(id(), outShape), graph()};
}

Shape Tensor::shape() const { return graph().shape(id()); }

Tensors Tensor::tensors(const TensorIds &ids_) const {
  Tensors tensors_;
  tensors_.reserve(ids_.size());
  for (auto i : ids_) {
    tensors_.push_back({i, graph()});
  }
  return tensors_;
}

std::string Tensor::opTypeString() const {
  return graph().typeString(opId());
}

void Tensor::setName(const std::string &dbs) const {
  graph().setName(opId(), dbs);
}

std::string Tensor::opName() const { return graph().getName(opId()); }

ConsumptionIds Tensor::consumptionIds() const {
  return graph().consumptionIds(id());
}

ConsumptionIds Tensor::modifiers() const { return graph().modifiers(id()); }

Tensors Tensor::allAliases() const {
  const auto ids = graph().allAliases(id());
  return tensors(ids);
}

bool Tensor::isAliasedTo(const Tensor &t) const {
  return graph().areAliased(id(), t.id());
}

bool Tensor::contains(const Tensor &t) const {
  return graph().contains(id(), t.id());
}

Tensor Tensor::pad(const std::array<std::vector<int64_t>, 2> &lu,
                   bool pw) const {
  return {graph().pad(id(), lu, pw), graph()};
}

Tensor Tensor::pad(const LowerPadding &l,
                   const UpperPadding &u,
                   ConstantPadding cp,
                   BroadcastPadding bp) const {
  return {graph().pad(id(), l, u, cp, bp), graph()};
}

Tensor Tensor::constant(const Shape &sh) const {
  return {graph().constant(sh), graph()};
}
Tensor Tensor::variable(const Shape &sh) const {
  return {graph().variable(sh), graph()};
}

std::ostream &operator<<(std::ostream &ost, const Tensor &t) {
  ost << '(';
  if (!t.graphName().empty()) {
    ost << t.graphName() << ',';
  }
  ost << t.id() << ')';
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const Tensors &ts) {
  util::append(ost, ts);
  return ost;
}

Tensor Tensor::variable(Graph &g, const Shape &shape) {
  return {g.variable(shape), g};
}
Tensor Tensor::constant(Graph &g, const Shape &shape) {
  return {g.constant(shape), g};
}

bool Tensor::aliasGateIsClosed() const {
  return graph().aliasGateIsClosed(opId());
}

Tensors Tensor::tensors(Graph &g, const TensorIds &ids) {
  Tensors ts_;
  ts_.reserve(ids.size());
  for (const auto &id : ids) {
    ts_.push_back({id, g});
  }
  return ts_;
}

std::string Tensor::graphName() const { return graph().getName(); }

} // namespace inplace
} // namespace memory
} // namespace poprithms
