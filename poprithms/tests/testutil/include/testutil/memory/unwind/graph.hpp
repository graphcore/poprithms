// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_TESTUTIL_UNWIND_TOY_GRAPH_HPP
#define POPRITHMS_TESTUTIL_UNWIND_TOY_GRAPH_HPP

#include "op.hpp"

#include <poprithms/common/schedulable/graph.hpp>
#include <poprithms/common/schedulable/op.hpp>
#include <poprithms/memory/unwind/scheduledsolution.hpp>
#include <poprithms/memory/unwind/sumlike.hpp>

namespace poprithms {
namespace unwindtoy {

using poprithms::common::multiout::InIndices;

class FullState;

/**
 * A graph class to test the lowering functionality of
 * poprithms::memory::unwind.
 *
 * It implemenents a small representative set of ops, more can be added as
 * required.
 * */
class Graph : public poprithms::common::schedulable::Graph {

public:
  Op::State getStartingState(const OpId opId,
                             const TensorIds &inIds,
                             const Shapes &outShapes);

  std::map<OpId, OpIds>
  schedulableDerivedSpecificConstraints(const OpIds &) const final {
    return {};
  }

  OpId insertOp(std::unique_ptr<Op> createdOp);

  poprithms::common::schedulable::SubGraphId singleGraph;

  template <class T, class... Args>
  OpId
  createOp(const TensorIds &inIds, const Shapes &outShapes, Args... args) {
    return insertOp(std::make_unique<T>(
        getStartingState(nOps_i64(), inIds, outShapes), args...));
  }

  ~Graph() override;

  Graph();

  Graph(const Graph &) = default;
  Graph(Graph &&)      = default;

  TensorId
  input(const Shape &s, double linear = 1.0, const std::string &name = {});

  TensorId slice(const TensorId &id, const Lower &l, const Upper &u) {
    return {createOp<Slice>({id}, {shape(id).slice(l, u)}, l, u), 0};
  }

  TensorId concat(const TensorIds &ids, uint64_t axis) {
    return {createOp<Concat>(ids, {Shape::concat(shapes(ids), axis)}, axis),
            0};
  }
  TensorId dimShuffle(const TensorId &id, const Permutation &p) {
    return {createOp<DimShuffle>({id}, {shape(id).dimShuffle(p)}, p), 0};
  }

  TensorId expand(const TensorId &id, const Shape &o) {
    return {createOp<Expand>({id}, {o}), 0};
  }

  MatMulAttractions matMulAttractions(OpId opId) const {
    return static_cast<const MatMul &>(op(opId)).atts_;
  }

  TensorId reduce(const TensorId &id, const Shape &s) {
    return {createOp<Reduce>({id}, {s}), 0};
  }

  TensorId sum(const TensorIds &ins,
               const std::vector<InIndex> &unwindIndices,
               const memory::unwind::SumAttractions &sats);

  // All the inputs with same shape as output are considered unwindable.
  TensorId sum(const TensorIds &inIds,
               const poprithms::memory::unwind::SumAttractions &satti);

  TensorId matmul(const TensorId &a,
                  const TensorId &b,
                  const MatMulAttractions &x = MatMulAttractions::Default()) {
    return {createOp<MatMul>({a, b}, {shape(a).matmul(shape(b))}, x), 0};
  }

  void appendOpColumns(std::ostream &ost, const OpIds &opIds) const final;
  bool multiOutTypeSpecificEqualTo(
      const poprithms::common::multiout::Graph &) const final;

  OpId insertBinBoundary(poprithms::common::schedulable::SubGraphId) final;

  const Op &op(OpId opId) const {
    return static_cast<const Op &>(multioutOp(opId));
  }

  // we won't be doing any substition in this test graph class.
  void schedulableTypeSpecificVerifyValidOutputSubstitute(
      const TensorId &,
      const TensorId &) const final {
    unimplemented();
  }

  void schedulableTypeSpecificRemoveOp(
      OpId,
      const poprithms::common::multiout::OptionalTensorIds &) final {
    unimplemented();
  }
};

std::ostream &operator<<(std::ostream &ost, const Graph &g);

} // namespace unwindtoy
} // namespace poprithms
#endif
