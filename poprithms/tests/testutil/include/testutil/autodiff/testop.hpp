// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef TESTUTIL_AUTODIFF_TESTOP_HPP
#define TESTUTIL_AUTODIFF_TESTOP_HPP

#include <poprithms/autodiff/ids/ids.hpp>

namespace poprithms {
namespace autodiff {
namespace test {

class Op {

public:
  enum class Type {
    Unknown = 0,
    UnknownGrad,
    Matmul,
    MatmulGrad,
    Add,
    Zero,
    Variable
  };

  // the gradient type of 't'.
  static Type grad(Type t);
  static std::string str(Type t);

  // a gradient flow.
  struct Flow {
    Flow(OutIndex o_, InIndex i_) : o(o_), i(i_) {}
    Flow(const OpTraversal &ot) : o(ot.outIndex()), i(ot.inIndex()) {}
    OutIndex o;
    InIndex i;
    bool operator==(const Flow &f) const { return o == f.o && i == f.i; }
  };

  Op(const TensorIds &ins_,
     uint64_t nOuts_,
     const std::vector<InIndex> &insRequired_,
     const std::vector<OutIndex> &outsRequired_,
     const std::vector<Flow> &,
     const std::string &name_ = {},
     Type                     = Type::Unknown);

  // The input tensors for the op
  TensorIds ins;

  // The number of outputs of the op
  uint64_t nOuts;

  // The consumers of each of the op's outputs
  std::vector<ConsumptionIds> consumers;

  // 1) To differentiate an op, which input tensors are required?
  std::vector<InIndex> insRequired;

  // 2) To differentiate an op, which output tensors are required?
  std::vector<OutIndex> outsRequired;

  // which outputs are differentiable w.r.t. to which inputs, with a non-zero
  // derivative?
  std::vector<Flow> flows;

  // name and type of this Op.
  std::string name;
  Type type;
};

} // namespace test
} // namespace autodiff
} // namespace poprithms

#endif
