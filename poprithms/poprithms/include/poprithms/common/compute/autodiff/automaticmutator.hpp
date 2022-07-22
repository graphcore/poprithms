// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef POPRITHMS_COMMON_COMPUTE_AUTODIFF_AUTOMATICMUTATOR_HPP
#define POPRITHMS_COMMON_COMPUTE_AUTODIFF_AUTOMATICMUTATOR_HPP

#include <memory>

#include <poprithms/autodiff/automatic/gradinfos.hpp>
#include <poprithms/autodiff/automatic/iautomaticmutator.hpp>
#include <poprithms/autodiff/automatic/requiredids.hpp>
#include <poprithms/autodiff/guide/graphinfo.hpp>
#include <poprithms/autodiff/ids/ids.hpp>
#include <poprithms/common/compute/autodiff/automaticquerier.hpp>
#include <poprithms/common/compute/autodiff/coregraphmutator.hpp>
#include <poprithms/common/compute/autodiff/guidegraphinfo.hpp>
#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/rtensor.hpp>
#include <poprithms/common/compute/scheduler.hpp>
#include <poprithms/common/compute/subgraph.hpp>
#include <poprithms/common/compute/tensor.hpp>

namespace poprithms {
namespace common {
namespace compute {

using poprithms::program::callstack::CarriedTensorIds;
using poprithms::program::callstack::IsStackedCopy;
using poprithms::program::callstack::StackedCopyOrder;

/**
 * Implementation of the IAutomaticMutator interface for a compute::Graph.
 * */
class AutomaticMutator
    : public poprithms::autodiff::automatic::IAutomaticMutator {

protected:
  Graph &graph_;

public:
  AutomaticMutator(Graph &m_) : graph_(m_) {}
  ~AutomaticMutator() override = default;

  TensorId concat_(const TensorIds &tIds, uint64_t dim) final {
    return Tensor::concat_(Tensor::tensors(tIds, graph_), dim);
  }

  SubGraphId subGraphId(OpId opId) const final {
    return graph_.subGraphId(opId);
  }

  TensorId scalarConstantLike(const TensorId &tId,
                              SubGraphId sgId,
                              double v,
                              const std::string &n) final {
    auto c0 = Tensor(tId, &graph_).constant(sgId, v);
    // set the name of c0 to n, and return it:
    return c0.name(n);
  }

  TensorId expand_(const TensorId &tId, const Shape &expanded) final {
    return Tensor(tId, &graph_).expand_(expanded);
  }

  TensorId broadcast_(const TensorId &tId, uint64_t N, uint64_t dim) final {
    return Tensor(tId, &graph_).broadcast_(N, dim);
  }

  TensorId reshape_(const TensorId &tId, const Shape &s) final {
    return Tensor(tId, &graph_).reshape_(s);
  }

  Shape shape(const TensorId &tId) const final { return graph_.shape(tId); }

  SubGraphId createSubGraphId(const std::string &n) final {
    return graph_.createSubGraphId(n);
  }

  TensorId zero_(const TensorId &tId) final {
    return Tensor(tId, &graph_).zero_();
  }

  TensorId variableLike(const TensorId &like,
                        SubGraphId sgId,
                        const std::string &n) final {
    return Tensor(like, &graph_).variable(sgId).name(n);
  }

  TensorId variableLike(const TensorId &like,
                        DType t,
                        const Shape &s,
                        const std::string &n) final {
    return Tensor(like, &graph_).variable(t, s).name(n);
  }

  TensorId add(const TensorId &a, const TensorId &b) final {
    return Tensor(a, &graph_).add(Tensor(b, &graph_));
  }

  void removeOp(OpId opId,
                const OptionalTensorIds &otis,
                const std::string &reason) final {
    graph_.removeOp(opId, otis, reason);
  }

  OpId
  switchOp(SubGraphId,
           const SubGraphIds &,
           const TensorId &,
           const std::vector<std::tuple<TensorId, TensorId, CalleeIndex>> &,
           const std::vector<std::vector<TensorId>> &,
           const std::vector<CalleeTensorIds> &) override {
    notImplemented("switchOp");
  }

  OpId call(SubGraphId caller,
            SubGraphId callee,
            const std::vector<std::pair<TensorId, TensorId>> &ins,
            const TensorIds &outs) override {
    return SubGraph(caller, graph_).call(callee, ins, outs);
  }

  virtual OpId
  repeat(SubGraphId caller,
         SubGraphId callee,
         uint64_t rptCount,
         const std::vector<std::pair<TensorId, TensorId>> &sis,
         const CarriedTensorIds &cis,
         const std::vector<std::pair<TensorId, IsStackedCopy>> &outs,
         StackedCopyOrder d) override {
    return SubGraph(caller, graph_)
        .repeat(callee, rptCount, sis, cis.carriedTensorIds(), outs, d);
  }

  TensorId encodeOneHot_(const TensorId &t, const TensorId &index) override {
    return Tensor(t, &graph_).encodeOneHot01_({index, &graph_});
  }

  [[noreturn]] void notImplemented(const std::string &x) const {
    throw poprithms::error::error("common::compute",
                                  x + " is not implemented. Coming soon.");
  }
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
