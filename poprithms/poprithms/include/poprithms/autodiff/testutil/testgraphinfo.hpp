// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef TESTUTIL_AUTODIFF_TESTGRAPH_HPP
#define TESTUTIL_AUTODIFF_TESTGRAPH_HPP

#include <poprithms/autodiff/guide/graphinfo.hpp>
#include <poprithms/autodiff/ids/ids.hpp>
#include <poprithms/autodiff/testutil/testop.hpp>

namespace poprithms {
namespace autodiff {
namespace testutil {

// Acts as actual graph as well as GraphInfo.
class TestGraphInfo : public poprithms::autodiff::guide::GraphInfo {

private:
  std::vector<Op> ops;

public:
  TestGraphInfo() = default;

  std::string str(const TensorId &tId) const { return tId.str(); }

  Op &op(OpId id) { return ops[id.get()]; }
  const Op &op(OpId id) const { return ops.at(id.get()); }
  OpId insert(Op op_);

  /// Insert an Op with inputs #ins_, where no gradient flows between any
  /// input or output.
  TensorId insertNoFlow(const TensorIds &ins_,
                        const std::string &n,
                        Op::Type = Op::Type::Unknown);

  /// Mimic a matmul in terms of the requirements for differention. There's no
  /// actual matrix multiplication here.
  OpId matmul(const TensorId &a, const TensorId &b, const std::string &n);

  bool gradientPropagates(const OpTraversal &ot) const final;

  void extendAutodiffRequiredTensors(OpId id,
                                     std::set<TensorId> &ids) const final;

  OpIds subSchedule(const std::set<OpId> &ids) const final {
    // Ops are inserted in topological order, and receive increasing OpIds. So
    // no kahn is required here.
    return OpIds(ids.cbegin(), ids.cend());
  }
  void appendOpInfo(std::ostream &, OpId) const final {}

  TensorIds inTensorIds(OpId id) const final { return op(id).ins; }

  TensorId inTensorId(OpId id, InIndex index) const final {
    return op(id).ins[index.get()];
  }
  uint64_t nInTensors(OpId id) const final { return op(id).ins.size(); }

  uint64_t nOutTensors(OpId id) const final { return op(id).nOuts; }

  ConsumptionIds consumptionIds(const TensorId &id) const final {
    return op(id.opId()).consumers.at(id.outIndex().get());
  }

  void assertCanBeRerun(OpId id, bool) const final;

  // this method is useful in projects where numerical types are used:
  // integral tensors can't have grads in general.
  void assertCanHaveGrad(const TensorId &) const final {}

  // this method is useful in projects where tensors can live in different
  // graphs. we won't be testing this here.
  void assertValidPaths(const TensorIds &, const TensorIds &) const final {}

  uint64_t nOps() const { return ops.size(); }

  bool isValueDependent(const OpTraversal &) const final { return true; }

  void append(std::ostream &) const;
};

std::ostream &operator<<(std::ostream &ost, const TestGraphInfo &);
std::ostream &operator<<(std::ostream &ost, const Op::Flow &);

} // namespace testutil
} // namespace autodiff
} // namespace poprithms

#endif
