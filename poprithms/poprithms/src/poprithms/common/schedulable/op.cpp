// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poprithms/common/multiout/graph.hpp>
#include <poprithms/common/schedulable/op.hpp>

namespace poprithms {
namespace common {
namespace schedulable {

void Op::insertControlDependencyOut(OpId ido) {
  if (std::find(controlDependencyOutOps_.cbegin(),
                controlDependencyOutOps_.cend(),
                ido) == controlDependencyOutOps_.cend()) {
    controlDependencyOutOps_.push_back(ido);
  }
}

void Op::removeControlDependencyOut(OpId id) {
  auto found = std::find(
      controlDependencyOutOps_.cbegin(), controlDependencyOutOps_.cend(), id);
  if (found != controlDependencyOutOps_.cend()) {
    controlDependencyOutOps_.erase(found);
  }
}

void Op::insertControlDependencyIn(OpId ido) {
  if (std::find(controlDependencyInOps_.cbegin(),
                controlDependencyInOps_.cend(),
                ido) == controlDependencyInOps_.cend()) {
    controlDependencyInOps_.push_back(ido);
  }
}

void Op::removeControlDependencyIn(OpId id) {
  auto found = std::find(
      controlDependencyInOps_.cbegin(), controlDependencyInOps_.cend(), id);
  if (found != controlDependencyInOps_.cend()) {
    controlDependencyInOps_.erase(found);
  }
}

bool Op::multiOutTypeSpecificEqualTo(
    const common::multiout::Op &other) const {

  // it is guaranteed that this is valid.
  const auto &rhs = static_cast<const Op &>(other);

  return
      // Same base properties:
      getState() == rhs.getState() &&

      // Same derived class:
      typeid(*this) == typeid(rhs) &&

      // Same derived class properties:
      schedulableTypeSpecificEqualTo(rhs);
}

Op::Op(const State &ob)
    : common::multiout::Op(ob.baseState), subGraphId_(ob.subGraphId),
      controlDependencyInOps_(ob.controlDependencyInOps),
      controlDependencyOutOps_(ob.controlDependencyOutOps) {}

bool Op::State::operator==(const State &rhs) const {
  return baseState == rhs.baseState &&                           //
         subGraphId == rhs.subGraphId &&                         //
         controlDependencyInOps == rhs.controlDependencyInOps && //
         controlDependencyOutOps == rhs.controlDependencyOutOps; //
}
} // namespace schedulable
} // namespace common
} // namespace poprithms
