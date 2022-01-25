// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef TESTUTIL_AUTODIFF_TESTGRAPHMUTATOR_HPP
#define TESTUTIL_AUTODIFF_TESTGRAPHMUTATOR_HPP

#include <testutil/autodiff/testgraphinfo.hpp>

#include <poprithms/autodiff/core/autodiff.hpp>

namespace poprithms {
namespace autodiff {
namespace test {

class TestGraphMutator : public core::GraphMutator {

public:
  TestGraphMutator(TestGraphInfo &c_) : GraphMutator(), c(c_) {}

private:
  TestGraphInfo &c;

  TensorId createZero(const TensorId &) final {
    return c.insertNoFlow({}, "", Op::Type::Zero);
  }

  TensorId createVariable(const TensorId &) final {
    return c.insertNoFlow({}, "", Op::Type::Variable);
  }

  // insert a clone of id into c
  OpId clone(OpId id, const TensorIds &ins) final;

  TensorId sum(const TensorIds &) final;
  // add(const TensorId &t0, const TensorId &t1) final;

  void setName(OpId id, const std::string &n) final { c.op(id).name = n; }

  // we always create just one grad op for a forward op. A forward op with
  // multiple (differentiable) inputs will generate a grad op with multiple
  // gradient outputs. This is not a constraint of the autodiff project, just
  // something which simplifies the testing.
  OptionalTensorIds getInGrads(OpId opId, const core::ToGradGraph &) final;
};

std::ostream &operator<<(std::ostream &, Op::Type);

} // namespace test
} // namespace autodiff
} // namespace poprithms

#endif
