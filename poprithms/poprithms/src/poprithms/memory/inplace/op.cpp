// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "op.hpp"

#include <memory>
#include <numeric>
#include <sstream>
#include <type_traits>

#include <memory/inplace/error.hpp>

#include <poprithms/common/multiout/graph.hpp>
#include <poprithms/common/multiout/util.hpp>
#include <poprithms/memory/alias/graph.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/memory/inplace/tensormap.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

bool Op::isViewOfAnyOutput(InIndex i) const {
  if (i.get() >= nInTensors()) {
    std::ostringstream oss;
    oss << "Invalid InIndex " << i << " for Op " << *this
        << ", which only has " << nInTensors() << " inputs. ";
    throw error(oss.str());
  }
  for (uint64_t o = 0; o < nOutTensors(); ++o) {
    if (isView(i, OutIndex(o))) {
      return true;
    }
  }
  return false;
}

Op::State::State(const OpId id_,
                 const TensorIds &inIds_,
                 const std::vector<ConsumptionIds> &consumptionIds_,
                 const Shapes &outShapes_,
                 const std::string &name_,
                 const OpIds &ins_,
                 const OpIds &outs_,
                 const Graph &g_)
    : State(common::multiout::Op::State(id_,
                                        inIds_,
                                        consumptionIds_,
                                        outShapes_,
                                        name_,
                                        g_),
            ins_,
            outs_) {}

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

void Op::grow(alias::Graph &g_, TensorMap &m) const {
  AliasTensorIds outIds = typeSpecificGrow(g_, m);
  for (uint64_t o = 0; o < nOutTensors(); ++o) {
    m.insert(outTensorId(o), outIds[o]);
  }
}

Op::State Op::getStartingState(const OpId opId,
                               const TensorIds &inIds,
                               const Shapes &outShapes,
                               const Graph &g) {

  const OpIds opOuts{};
  const std::string name{};

  // No consumptionIds at any of the output indices.
  const std::vector<ConsumptionIds> consumptionIds(outShapes.size());

  return State(opId,
               inIds,
               consumptionIds,
               outShapes,
               name,
               common::multiout::Graph::opIds(inIds),
               opOuts,
               g);
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
