// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/memory/alias/node.hpp>

namespace poprithms {
namespace memory {
namespace alias {

void Node::insertOut(TensorId ido) {
  if (std::find(outs_.cbegin(), outs_.cend(), ido) == outs_.cend()) {
    outs_.push_back(ido);
  }
}

std::vector<TensorId> Node::insAndOuts() const {
  std::vector<TensorId> insAndOuts_;
  insAndOuts_.reserve(ins_.size() + outs_.size());
  insAndOuts_.insert(insAndOuts_.end(), ins_.cbegin(), ins_.cend());
  insAndOuts_.insert(insAndOuts_.end(), outs_.cbegin(), outs_.cend());
  return insAndOuts_;
}

bool Node::operator==(const Node &rhs) const {
  return ins_ == rhs.ins_ && outs_ == rhs.outs_ &&
         inShapes_ == rhs.inShapes_ && id_ == rhs.id_ &&
         typeString() == rhs.typeString();
}

} // namespace alias
} // namespace memory
} // namespace poprithms
