// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "op.hpp"

#include <memory>
#include <numeric>
#include <sstream>
#include <type_traits>

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
      consumers_(ob.consumers), outShapes_(ob.outShapes), name_(ob.name) {}

// C++20 we will be able to just use (= default).
bool Op::State::operator==(const State &rhs) const {
  return                            //
      id == rhs.id &&               //
      ins == rhs.ins &&             //
      outs == rhs.outs &&           //
      inIds == rhs.inIds &&         //
      consumers == rhs.consumers && //
      outShapes == rhs.outShapes && //
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

void Op::grow(alias::Graph &g, TensorMap &m) const {
  AliasTensorIds outIds = typeSpecificGrow(g, m);
  for (uint64_t o = 0; o < nOutTensors(); ++o) {
    m.insert(outTensorId(o), outIds[o]);
  }
}

Op::State Op::getBaseState(const OpId opId,
                           const TensorIds &inIds,
                           const Shapes &outShapes,
                           const OpIds &opIns) {

  const OpIds opOuts{};
  const std::string name{};

  // No consumers at any of the output indices.
  std::vector<Consumers> consumers(outShapes.size());

  return State(opId, opIns, opOuts, inIds, consumers, outShapes, name);
}

} // namespace inplace
} // namespace memory
} // namespace poprithms
