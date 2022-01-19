// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <testutil/memory/unwind/fullstate.hpp>
#include <testutil/memory/unwind/graph.hpp>

#include <poprithms/common/multiout/consumptionid.hpp>
#include <poprithms/common/multiout/graph.hpp>
#include <poprithms/common/schedulable/graph.hpp>
#include <poprithms/common/schedulable/op.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/memory/unwind/lower.hpp>
#include <poprithms/memory/unwind/scheduledsolution.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/permutation.hpp>

namespace poprithms {
namespace unwindtoy {

void FullState::unwindAndUpdate(const Path &p,
                                const HTensor &src,
                                const HTensor &dst) const {

  auto unwound = p.chain().apply(src);
  for (const auto &r : p.dstRegions().get()) {
    const auto tileMapping = unwound.gather(r.getOns());
    dst.gather_(r.getOns()).update_(tileMapping);
  }
}

class Translator : public poprithms::memory::unwind::Translator {
public:
  const FullState &uw;
  const Graph &g_;
  Translator(const FullState &u, const Graph &g) : uw(u), g_(g) {}
  TensorId fromUnwind(const TensorId &uwId) const { return uw.toToy(uwId); }
  std::string str(OpId xtId) const { return g_.op(xtId).str(); }
};

void FullState::initialize(OpId opId) { tg_.op(opId).fwd(*this); }

void FullState::lower() {
  ssp = std::make_unique<ScheduledSolution>(
      uwg_, Translator(*this, tg_), tg_.getForwardEdgeMap_u64());

  poprithms::memory::unwind::Lowerer<HTensor, FullState>::lower(*this);
}

HTensor FullState::mainLayout(const TensorId &toyId) const {
  return mainLayouts_.at(toyId);
}

bool FullState::unwindSinkInitialized(const TensorId &tId) const {
  return (unwindSinks_.find(tId) != unwindSinks_.cend());
}

void FullState::initializeUnwindSink(const TensorId &pathDst) {
  unwindSinks_.insert({pathDst, createUnmapped(uwg_.shape(pathDst))});
}

HTensor FullState::getUnwindSink(const TensorId &pathDst) const {
  return unwindSinks_.at(pathDst);
}

void FullState::setMainLayout(const TensorId &toyId, const HTensor &ht) {
  mainLayouts_.insert({toyId, ht});
}

OpId FullState::unwindOpWithName(
    const std::vector<std::string> &frags) const {

  OpIds found;

  auto sat = [this, &frags](OpId opId) {
    auto name_ = uwg_.getName(opId);
    for (auto f : frags) {
      if (name_.find(f) == std::string::npos) {
        return false;
      }
    }
    return true;
  };

  for (auto opId : uwg_.opIds()) {
    if (sat(opId)) {
      found.push_back(opId);
    }
  }

  if (found.size() == 1) {
    return found[0];
  }

  std::ostringstream oss;

  if (found.empty()) {
    oss << "No ";
  } else {
    oss << "Multiple ";
  }
  oss << "Ops in the unwind::Graph with the frags, ";
  poprithms::util::append(oss, frags);
  throw poprithms::test::error(oss.str());
}

FullState::FullState(const Graph &g) : tg_(g) {
  for (auto opId : g.vanillaSchedule()) {
    g.op(opId).growUnwind(*this);
  }
}

TensorId FullState::toUnwind(const TensorId &id) const {
  auto uw = toUnwind_.find(id);
  if (uw == toUnwind_.cend()) {
    throw poprithms::test::error("No unwind id for " + id.str());
  }
  return uw->second;
}
TensorIds FullState::toUnwinds(const TensorIds &ids) const {
  TensorIds uws;
  for (auto id : ids) {
    uws.push_back(toUnwind(id));
  }
  return uws;
}
TensorId FullState::toToy(const TensorId &tId) const {

  auto t = toToy_.find(tId);
  if (t == toToy_.cend()) {
    throw poprithms::test::error("No toy id for " + tId.str());
  }
  return t->second;
}

void FullState::insert(const TensorId &toy, const TensorId &uw) {
  toUnwind_.insert({toy, uw});
  toToy_.insert({uw, toy});
}

HTensor FullState::createUnmapped(const Shape &s) const {
  return HTensor::float32(unmappedValue).expand(s);
}

HTensor FullState::createUnmapped(const Path &, const Shape &s) const {
  return createUnmapped(s);
}

HTensor FullState::createMappedSrc(const Path &p, const HTensors &ins) const {
  return createMappedSrc(p.src());
}

HTensor FullState::createMappedSrc(const TensorId &uwId) const {
  return HTensor::uniformFloat32(-1, 1, uwg_.shape(uwId), uwId.opId().get());
}

std::pair<bool, HTensor> FullState::finalLayout(const TensorId &uwId) const {

  // The tensor in the unwind graph has no corresponding tensor in the toy
  // (ML) graph:
  if (toToy_.find(uwId) == toToy_.cend()) {
    return {false, createEmpty()};
  }

  auto f0 = mainLayouts_.find(toToy(uwId));
  // The tensor in the unwind graph does have a corresponding tensor in the
  // toy (ML) graph, but the toy tensor has not been allocated a final layout
  // yet:
  if (f0 == mainLayouts_.cend()) {
    return {false, createEmpty()};
  }

  if (f0->second.shape() != tg_.shape(toToy(uwId))) {
    std::ostringstream oss;
    oss << "Error in FullState::finalLayout(uwId = " << uwId
        << "). There is a shape mismatch. ";
    throw poprithms::test::error(oss.str());
  }
  return {true, f0->second};
}

} // namespace unwindtoy
} // namespace poprithms
