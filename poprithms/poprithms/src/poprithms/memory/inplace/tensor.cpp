// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "ops.hpp"

#include <poprithms/memory/alias/graph.hpp>
#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

TensorIds Tensor::tensorIds(const Tensors &tensors) {
  TensorIds ids_;
  ids_.reserve(tensors.size());
  for (const auto &t : tensors) {
    ids_.push_back(t.id());
  }
  return ids_;
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

Tensor Tensor::mux(const Tensors &ts, InIndex inIndex) {
  if (ts.empty()) {
    throw error("Cannot create Mux with 0 inputs");
  }
  return {ts[0].graph().mux(tensorIds(ts), inIndex), ts[0].graph()};
}

Tensor Tensor::mux(const Tensors &ts) {
  if (ts.empty()) {
    throw error("Cannot create Mux with 0 inputs");
  }
  return {ts[0].graph().mux(tensorIds(ts)), ts[0].graph()};
}

Tensors Tensor::multi(Graph &g,
                      const Tensors &ins,
                      const Shapes &outs,
                      const CrossAliases &cas) {

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

Tensor Tensor::mux(bool isOpen) const {
  if (!isOpen) {
    return {graph().mux({id()}), graph()};
  }
  return {graph().mux({id()}, 0), graph()};
}

Tensor Tensor::unary() const { return {graph().unary(id()), graph()}; }

Tensor Tensor::reshape(const Shape &outShape) const {
  return {graph().reshape(id(), outShape), graph()};
}

Tensor Tensor::flatten() const { return reshape(shape().flatten()); }

Tensor Tensor::subSample(int64_t stride, uint64_t dimension) const {
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

void Tensor::setName(const std::string &dbs) { graph().setName(opId(), dbs); }

Consumers Tensor::consumers() const { return graph().consumers(id()); }

Consumers Tensor::modifiers() const { return graph().modifiers(id()); }

Tensors Tensor::allAliases() const {
  const auto ids = graph().allAliases(id());
  return tensors(ids);
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

bool Tensor::muxIsClosed() const { return graph().muxIsClosed(opId()); }

Tensors Tensor::tensors(Graph &g, const TensorIds &ids) {
  Tensors ts_;
  ts_.reserve(ids.size());
  for (const auto &id : ids) {
    ts_.push_back({id, g});
  }
  return ts_;
}

std::string Tensor::graphName() const { return graph().name(); }

} // namespace inplace
} // namespace memory
} // namespace poprithms
