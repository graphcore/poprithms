// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory>
#include <numeric>
#include <sstream>
#include <type_traits>

#include <memory/unwind/error.hpp>
#include <memory/unwind/op.hpp>

#include <poprithms/memory/alias/graph.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

const std::vector<ValuedTensorIds> &Op::valuedPartners() const {
  return valuedPartners_;
}

ValuedTensorIds Op::valuedPartners(OutIndex outIndex) const {
  return valuedPartners_[outIndex.get()];
}

Op::State Op::getState() const {
  return State{common::multiout::Op::getState(), valuedPartners_};
}

DisjointRegions
Op::outRegions(const DisjointRegions &inRegs, InIndex i, OutIndex o) const {
  auto ch = Chain(inShape(i));
  extendFwd(ch, i, o);
  return ch.apply(inRegs);
}

DisjointRegions
Op::inRegions(const DisjointRegions &inRegs, InIndex i, OutIndex o) const {
  auto ch = Chain(outShape(o));
  extendBwd(ch, i, o);
  return ch.apply(inRegs);
}

Op::~Op() = default;

Op::State Op::getStartingState(const OpId opId,
                               const TensorIds &inIds,
                               const Shapes &inShapes,
                               const Shapes &outShapes) {

  const std::string name{};

  // No consumptionIds at any of the output indices.
  const std::vector<ConsumptionIds> consumptionIds(outShapes.size());

  const std::vector<ValuedTensorIds> valuedPartners__(outShapes.size());

  return State(opId,
               inIds,
               consumptionIds,
               inShapes,
               outShapes,
               name,
               valuedPartners__);
}

void Op::extend(Chain &c, InIndex i, OutIndex o, bool isFwd) const {
  if (isFwd) {
    extendFwd(c, i, o);
  } else {
    extendBwd(c, i, o);
  }
}

void Op::insertAttractor(OutIndex oi, const TensorId &dst, double val) {

  // If there is already an attraction to dst, increase its value.
  for (auto &att : valuedPartners_[oi.get()]) {
    if (att.tensorId() == dst) {
      att.setValue(att.value() + val);
      return;
    }
  }
  valuedPartners_[oi.get()].push_back({dst, val});
}

// C++20 we will be able to just use (= default).
bool Op::State::operator==(const State &rhs) const {
  return                                    //
      baseState == rhs.baseState &&         //
      valuedPartners == rhs.valuedPartners; //
}

bool Op::multiOutTypeSpecificEqualTo(const common::multiout::Op &rhs_) const {

  const auto &rhs = static_cast<const Op &>(rhs_);
  return
      // Same base properties:
      getState() == rhs.getState() &&
      // Same derived class:
      typeid(*this) == typeid(rhs) &&
      // Same derived class properties:
      unwindTypeSpecificEqualTo(rhs);
}

std::ostream &operator<<(std::ostream &os, const Op &op) {
  os << op.str();
  if (!op.getName().empty()) {
    os << "::" << op.getName();
  }
  return os;
}

std::vector<InIndex> Op::unwindableIndices(OutIndex o) const {
  std::vector<InIndex> inIndices;
  for (uint64_t i = 0; i < nInTensors(); ++i) {
    if (isUnwindable(i, o)) {
      inIndices.push_back(i);
    }
  }
  return inIndices;
}

std::vector<OutIndex> Op::unwindableIndices(InIndex i) const {
  std::vector<OutIndex> outIndices;
  for (uint64_t o = 0; o < nOutTensors(); ++o) {
    if (isUnwindable(i, o)) {
      outIndices.push_back(o);
    }
  }
  return outIndices;
}

} // namespace unwind
} // namespace memory
} // namespace poprithms
