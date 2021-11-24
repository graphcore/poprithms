// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <testutil/autodiff/testop.hpp>

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace autodiff {
namespace test {

std::ostream &operator<<(std::ostream &os, Op::Type t) {
  os << Op::str(t);
  return os;
}

Op::Op(const TensorIds &ins_,
       const uint64_t nOuts_,
       const std::vector<InIndex> &insRequired_,
       const std::vector<OutIndex> &outsRequired_,
       const std::vector<Flow> &flows_,
       const std::string &name_,
       Type t)
    : ins(ins_), nOuts(nOuts_), insRequired(insRequired_),
      outsRequired(outsRequired_), flows(flows_), name(name_), type(t) {
  consumers.resize(nOuts);
}

std::string Op::str(Op::Type t) {
  switch (t) {
  case Op::Type::Unknown:
    return "Unknown";
  case Op::Type::UnknownGrad:
    return "UnknownGrad";
  case Op::Type::Matmul:
    return "Matmul";
  case Op::Type::MatmulGrad:
    return "MatmulGrad";
  case Op::Type::Add:
    return "Add";
  case Op::Type::Zero:
    return "Zero";
  case Op::Type::Variable:
    return "Variable";
  }
  throw poprithms::test::error("Unhandled type in Op::str");
}

Op::Type Op::grad(Type t) {

  switch (t) {
  case Op::Type::Unknown:
    return Op::Type::UnknownGrad;
  case Op::Type::Matmul:
    return Op::Type::MatmulGrad;
  default: {
    throw poprithms::test::error("Unhandled type in Op::grad, " + str(t));
  }
  }
  throw poprithms::test::error("Unhandled type in Op::grad, " + str(t));
}

} // namespace test

} // namespace autodiff
} // namespace poprithms
