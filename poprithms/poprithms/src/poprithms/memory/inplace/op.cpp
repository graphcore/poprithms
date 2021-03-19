// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "op.hpp"

#include <memory>
#include <numeric>
#include <sstream>
#include <type_traits>

#include <poprithms/common/multiout/graph.hpp>
#include <poprithms/common/multiout/util.hpp>
#include <poprithms/memory/alias/graph.hpp>
#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/tensormap.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

Op::~Op() = default;

bool Op::modifies() const {
  for (uint64_t i = 0; i < nInTensors(); ++i) {
    if (modifies(i)) {
      return true;
    }
  }
  return false;
}

std::vector<InIndex> Op::modifyingIndices() const {
  std::vector<InIndex> indices_;
  for (uint64_t i = 0; i < nInTensors(); ++i) {
    if (modifies(i)) {
      indices_.push_back(i);
    }
  }
  return indices_;
}

void Op::insertOut(OpId ido) {
  if (std::find(outs_.cbegin(), outs_.cend(), ido) == outs_.cend()) {
    outs_.insert(std::upper_bound(outs_.cbegin(), outs_.cend(), ido), ido);
  }
}

void Op::insertIn(OpId ido) {
  if (std::find(ins_.cbegin(), ins_.cend(), ido) == ins_.cend()) {
    ins_.insert(std::upper_bound(ins_.cbegin(), ins_.cend(), ido), ido);
  }
}

// C++20 we will be able to just use (= default).
bool Op::State::operator==(const State &rhs) const {
  return                            //
      baseState == rhs.baseState && //
      ins == rhs.ins &&             //
      outs == rhs.outs              //
      ;
}

std::ostream &operator<<(std::ostream &os, const Op &op) {
  os << op.str();
  if (!op.getName().empty()) {
    os << "::" << op.getName();
  }
  return os;
}

void Op::grow(alias::Graph &g, TensorMap &m) const {
  AliasTensorIds outIds = typeSpecificGrow(g, m);
  for (uint64_t o = 0; o < nOutTensors(); ++o) {
    m.insert(outTensorId(o), outIds[o]);
  }
}

Op::State Op::getStartingState(const OpId opId,
                               const TensorIds &inIds,
                               const Shapes &inShapes,
                               const Shapes &outShapes) {

  const OpIds opOuts{};
  const std::string name{};

  // No consumptionIds at any of the output indices.
  const std::vector<ConsumptionIds> consumptionIds(outShapes.size());

  return State(opId,
               inIds,
               consumptionIds,
               inShapes,
               outShapes,
               name,
               common::multiout::Graph::opIds(inIds),
               opOuts);
}

bool Op::multiOutTypeSpecificEqualTo(const common::multiout::Op &rhs_) const {

  const auto &rhs = static_cast<const Op &>(rhs_);
  return
      // Same base properties:
      getState() == rhs.getState() &&
      // Same derived class:
      typeid(*this) == typeid(rhs) &&
      // Same derived class properties:
      inplaceTypeSpecificEqualTo(rhs);
}

} // namespace inplace
} // namespace memory
} // namespace poprithms
