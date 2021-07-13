// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poprithms/common/multiout/graph.hpp>
#include <poprithms/common/schedulable/op.hpp>

namespace poprithms {
namespace common {
namespace schedulable {

void Op::insertOut(OpId ido) {
  if (std::find(outOps_.cbegin(), outOps_.cend(), ido) == outOps_.cend()) {
    outOps_.push_back(ido);
  }
}

void Op::insertIn(OpId ido) {
  if (std::find(inOps_.cbegin(), inOps_.cend(), ido) == inOps_.cend()) {
    inOps_.push_back(ido);
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

namespace {
OpIds setDifference(const OpIds &a, const OpIds &b) {
  // c = a\b.
  OpIds c;
  const auto nToReserve = a.size() > b.size() ? a.size() - b.size() : 0;
  c.reserve(nToReserve);
  for (auto id : a) {
    if (std::find(b.cbegin(), b.cend(), id) == b.cend()) {
      c.push_back(id);
    }
  }
  return c;
}
} // namespace

OpIds Op::nonDataInOps() const {
  auto inDataDeps_ = poprithms::common::multiout::Graph::opIds(inTensorIds());
  return setDifference(inOps(), inDataDeps_);
}

Op::Op(const State &ob)
    : common::multiout::Op(ob.baseState), subGraphId_(ob.subGraphId),
      inOps_(ob.inOps), outOps_(ob.outOps) {}

bool Op::State::operator==(const State &rhs) const {
  return baseState == rhs.baseState &&   //
         subGraphId == rhs.subGraphId && //
         inOps == rhs.inOps &&           //
         outOps == rhs.outOps;           //
}
} // namespace schedulable
} // namespace common
} // namespace poprithms
