// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <memory>
#include <numeric>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

#include <autodiff/autodiff/error.hpp>

#include <poprithms/autodiff/automatic/gradinfos.hpp>
#include <poprithms/common/schedulable/graph.hpp>

namespace poprithms {
namespace autodiff {
namespace automatic {

SubGraphIds
GradInfos::gradGraphsCreatedFor(const Objective &objective) const {
  auto found = gradsForObjective_.find(objective);
  if (found == gradsForObjective_.end()) {
    return {};
  }
  return found->second;
}

void GradInfos::insert(SubGraphId sgId, const GradInfo &inf) {

  if (inf.gradSubGraphId() != sgId) {
    std::ostringstream oss;
    oss << "The sub-graph " << sgId
        << " should be the gradient graph for the GradInfo being inserted, "
        << "but it is not. The gradient sub-graph of the GradInfo is "
        << inf.gradSubGraphId();
    throw error(oss.str());
  }

  {
    auto found = gradInfos_.find(sgId);
    if (found != gradInfos_.cend()) {
      std::ostringstream oss;
      oss << "There is already gradient information for the sub-graph "
          << sgId.str() << ", it can only be gradient of 1 graph+objective. ";
      throw error(oss.str());
    }
    gradInfos_.insert({sgId, inf});
  }

  {
    auto found = gradsForObjective_.find(inf.objective());
    if (found == gradsForObjective_.cend()) {
      gradsForObjective_.insert({inf.objective(), {sgId}});
    } else {
      found->second.push_back(sgId);
    }
  }
}

const GradInfo &GradInfos::at(SubGraphId gradSgId) const {
  const auto found = gradInfos_.find(gradSgId);
  if (found == gradInfos_.cend()) {
    std::ostringstream oss;
    oss << "No GradInfo found for the (gradient) sub-graph " << gradSgId;
    throw error(oss.str());
  }
  return found->second;
}

SubGraphId GradInfos::grad(OpId opId, CalleeIndex ci) const {

  auto errm = [opId, ci]() {
    std::ostringstream oss;
    oss << "There is no gradient registered for callee #" << ci << " of op "
        << opId;
    return oss.str();
  };
  auto found = gradsForCallees.find(opId);
  if (found == gradsForCallees.cend()) {
    throw error(errm());
  }
  if (found->second.size() <= ci.get()) {
    throw error(errm());
  }
  return found->second.at(ci.get());
}

void GradInfos::setGrad(OpId opId, CalleeIndex ci, SubGraphId sgId) {
  auto found = gradsForCallees.find(opId);
  if (found == gradsForCallees.cend()) {
    SubGraphIds sgIds(ci.get() + 1, SubGraphId::unset());
    sgIds.at(ci.get()) = sgId;
    gradsForCallees.insert({opId, sgIds});
  } else {
    if (found->second.size() <= ci.get()) {
      found->second.resize(ci.get() + 1, SubGraphId::unset());
      found->second.at(ci.get()) = sgId;
    }
  }
}

bool GradInfos::hasGrad(OpId opId, CalleeIndex ci) const {
  auto found = gradsForCallees.find(opId);

  // If the op has no grads for callees registered, at all:
  if (found == gradsForCallees.cend()) {
    return false;
  }

  // If the op has grads registered for callees, but not all the wayi up to
  // #ci:
  if (found->second.size() <= ci.get()) {
    return false;
  }

  return !found->second.at(ci.get()).isUnset();
}

} // namespace automatic
} // namespace autodiff
} // namespace poprithms
