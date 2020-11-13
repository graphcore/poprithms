// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <poprithms/outline/linear/error.hpp>
#include <poprithms/outline/linear/op.hpp>
#include <poprithms/outline/linear/tensor.hpp>

namespace poprithms {
namespace outline {
namespace linear {

namespace {
const TensorId NoTensorId(std::numeric_limits<uint64_t>::max());
}

void Op::insertIn(TensorId x, InIndex i) {
  if (ins_.size() < i + 1) {
    ins_.resize(i + 1, NoTensorId);
  } else if (ins_[i] != NoTensorId) {
    std::ostringstream oss;
    oss << "There is already an input at index " << i << " for Op "
        << debugStr();
    throw error(oss.str());
  }
  ins_[i] = x;
}

void Op::insertOut(TensorId x, OutIndex i) {
  if (outs_.size() < i + 1) {
    outs_.resize(i + 1, NoTensorId);
  } else if (outs_[i] != NoTensorId) {
    std::ostringstream oss;
    oss << "There is already an output at index " << i << " for Op "
        << debugStr();
    throw error(oss.str());
  }
  outs_[i] = x;
}

void Op::append(std::ostream &ost) const {
  ost << "debugStr:" << debugStr() << "  id:" << id() << "  color:" << color()
      << "  type:" << type();
}

std::ostream &operator<<(std::ostream &ost, const Op &op) {
  op.append(ost);
  return ost;
}

} // namespace linear
} // namespace outline

} // namespace poprithms
