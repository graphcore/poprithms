// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory/alias/error.hpp>

#include <poprithms/memory/alias/node.hpp>

namespace poprithms {
namespace memory {
namespace alias {

Shapes Node::State::inShapes() const {
  std::vector<Shape> shapes;
  shapes.reserve(ins.size());
  for (auto i : ins) {
    shapes.push_back(graph.shape(i));
  }
  return shapes;
}

void Node::insertOut(TensorId ido) {
  if (std::find(outs_.cbegin(), outs_.cend(), ido) == outs_.cend()) {
    outs_.push_back(ido);
  }
}

void Node::removeOut(TensorId ido) {
  auto found = std::find(outs_.cbegin(), outs_.cend(), ido);
  outs_.erase(found);
}

std::vector<TensorId> Node::insAndOuts() const {
  std::vector<TensorId> insAndOuts_;
  insAndOuts_.reserve(ins_.size() + outs_.size());
  insAndOuts_.insert(insAndOuts_.end(), ins_.cbegin(), ins_.cend());
  insAndOuts_.insert(insAndOuts_.end(), outs_.cbegin(), outs_.cend());
  return insAndOuts_;
}

bool Node::operator==(const Node &rhs) const {
  return ins_ == rhs.ins_ && outs_ == rhs.outs_ && id_ == rhs.id_ &&
         typeString() == rhs.typeString();
}

void Node::noWeakVTables() { throw error(error::error::weakVTableMessage()); }

} // namespace alias
} // namespace memory
} // namespace poprithms
