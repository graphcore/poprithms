// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "op.hpp"

#include <memory>
#include <numeric>
#include <sstream>
#include <type_traits>

#include <poprithms/memory/alias/graph.hpp>
#include <poprithms/memory/inplace/aliastype.hpp>
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

void Op::apply(alias::Graph &g, TensorMap &m, AliasType t) {
  if (t.isOutplace()) {
    applyOutplaceTo(g, m);
  } else {
    applyInplaceTo(g, m, t);
  }
  aType_ = t;
}

void Op::insertOut(OpId ido) {
  if (std::find(outs_.cbegin(), outs_.cend(), ido) == outs_.cend()) {
    outs_.insert(std::upper_bound(outs_.cbegin(), outs_.cend(), ido), ido);
  }
}

void Op::insertConsumer(OutIndex outIndex, const Consumer &consumer) {
  consumers_[outIndex.get()].push_back(consumer);
  insertOut(consumer.opId());
}

void Op::insertIn(OpId ido) {
  if (std::find(ins_.cbegin(), ins_.cend(), ido) == ins_.cend()) {
    ins_.insert(std::upper_bound(ins_.cbegin(), ins_.cend(), ido), ido);
  }
}

Op::Op(const State &ob)
    : id_(ob.id), ins_(ob.ins), outs_(ob.outs), inIds_(ob.inIds),
      consumers_(ob.consumers), outShapes_(ob.outShapes), aType_(ob.aType),
      name_(ob.name) {}

// C++20 we will be able to just use (= default).
bool Op::State::operator==(const State &rhs) const {
  return                            //
      id == rhs.id &&               //
      ins == rhs.ins &&             //
      outs == rhs.outs &&           //
      inIds == rhs.inIds &&         //
      consumers == rhs.consumers && //
      outShapes == rhs.outShapes && //
      aType == rhs.aType &&         //
      name == rhs.name;
}

TensorIds Op::outTensorIds() const {
  TensorIds outIds;
  outIds.reserve(nOutTensors());
  for (uint64_t o = 0; o < nOutTensors(); ++o) {
    outIds.push_back(TensorId{id(), OutIndex(o)});
  }
  return outIds;
}

std::ostream &operator<<(std::ostream &os, const Op &op) {
  os << op.str();
  return os;
}

TensorIds Op::inAliasIdsIf(AliasType t) const {
  const auto inIndices = inAliasIndicesIf(t);
  TensorIds inIds;
  inIds.reserve(inIndices.size());
  for (auto inIndex : inIndices) {
    inIds.push_back(inTensorId(inIndex));
  }
  return inIds;
}

TensorIds Op::inModifiedIdsIf(AliasType t) const {
  const auto inIndices = inModifiedIndicesIf(t);
  TensorIds inIds;
  inIds.reserve(inIndices.size());
  for (auto inIndex : inIndices) {
    inIds.push_back(inTensorId(inIndex));
  }
  return inIds;
}

TensorIds Op::outAliasIdsIf(AliasType t) const {
  const auto outIndices = outAliasIndicesIf(t);
  TensorIds outIds;
  outIds.reserve(outIndices.size());
  for (auto outIndex : outIndices) {
    outIds.push_back(outTensorId(outIndex));
  }
  return outIds;
}

void Op::grow(alias::Graph &g, TensorMap &m) const {
  AliasTensorIds outIds = typeSpecificGrow(g, m);
  for (uint64_t o = 0; o < nOutTensors(); ++o) {
    m.insert(outTensorId(o), outIds[o]);
  }
}

void Op::verifyAllInplace(AliasType t) const {
  if (!t.isAllInplace()) {
    std::ostringstream oss;
    oss << "error in Op::verifyAllInplace, with AliasType " << t;
    throw error(oss.str());
  }
}

} // namespace inplace
} // namespace memory
} // namespace poprithms
