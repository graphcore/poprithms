// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <sstream>

#include <common/compute/error.hpp>

#include <poprithms/autodiff/automatic/requiredids.hpp>
#include <poprithms/common/compute/autodiff/guidegraphinfo.hpp>

namespace poprithms {
namespace common {
namespace compute {

void GuideGraphInfo::extendAutodiffRequiredTensors(
    OpId id,
    std::set<TensorId> &s) const {

  autodiff::automatic::RequiredIds mar(s, gradInfos);
  graph_.computeOp(id).extendAutodiffRequiredTensors(mar);
}

void GuideGraphInfo::assertCanBeRerun(OpId opId) const {
  if (graph_.isVarInit(opId)) {
    std::ostringstream oss;
    oss << "\nFailure in assertCanBeRerun for "
        << "the VarInit op with OpId " << opId << ":\n";
    graph_.appendOpColumns(oss, {opId});
    oss << ",\n. This might be caused by backtracking "
        << "too far due to insufficient checkpointing. ";
    throw error(oss.str());
  }
}

void GuideGraphInfo::assertCanHaveGrad(const TensorId &tId) const {
  if (ndarray::isFixedPoint(graph_.dtype(tId))) {
    std::ostringstream oss;
    oss << "A fixed-point tensor cannot have a gradient. ";
    oss << "The creator of the fixed-point Tensor " << tId << " is\n";
    graph_.appendOpColumns(oss, {tId.opId()});
    throw error(oss.str());
  }
}

void GuideGraphInfo::assertValidPaths(
    const TensorIds &targets,
    const TensorIds &gradsProvidedFor) const {

  auto targetSubGraphIds           = graph_.subGraphIds(targets);
  auto gradsProvidedForSubGraphIds = graph_.subGraphIds(gradsProvidedFor);

  auto allSubGraphIds = targetSubGraphIds;
  allSubGraphIds.insert(allSubGraphIds.end(),
                        gradsProvidedForSubGraphIds.cbegin(),
                        gradsProvidedForSubGraphIds.cend());

  if (allSubGraphIds.size() >= 2) {
    for (const auto &sgId : allSubGraphIds) {
      if (sgId != allSubGraphIds[0]) {
        std::ostringstream oss;
        oss << "Targets and gradients provided for must be in the same "
               "sub-graph. "
            << "At least 2 sub-graphs observed: " << sgId << " and "
            << allSubGraphIds.at(0);
        throw error(oss.str());
      }
    }
  }
}

} // namespace compute
} // namespace common
} // namespace poprithms
