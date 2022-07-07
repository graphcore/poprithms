// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <sstream>

#include <autodiff/autodiff/error.hpp>

#include <poprithms/autodiff/guide/objective.hpp>
#include <poprithms/util/unisort.hpp>

namespace poprithms {
namespace autodiff {
namespace guide {

TensorIds Objective::allTensorIds() const {
  TensorIds x = gradsProvided_;
  x.insert(x.end(), checkpoints_.cbegin(), checkpoints_.cend());
  x.insert(x.end(), targets_.cbegin(), targets_.cend());
  x.insert(x.end(), gradsProvidedFor_.cbegin(), gradsProvidedFor_.cend());
  return x;
}

bool Objective::isCheckpoint(const TensorId &inNonGrad) const {
  return std::find(checkpoints_.cbegin(), checkpoints_.cend(), inNonGrad) !=
         checkpoints_.cend();
}

bool Objective::hasGradProvided(const TensorId &inNonGrad) const {
  return std::find(gradsProvidedFor_.cbegin(),
                   gradsProvidedFor_.cend(),
                   inNonGrad) != gradsProvidedFor_.cend();
}

bool Objective::isTarget(const TensorId &inNonGrad) const {
  return std::find(targets_.cbegin(), targets_.cend(), inNonGrad) !=
         targets_.cend();
}

Objective::Objective(const TensorIds &gradsProvidedFor,
                     const TensorIds &checkpoints,
                     const TensorIds &targets,
                     InGraph inGraph,
                     const TensorIds &gradsProvided)
    : gradsProvidedFor_(gradsProvidedFor), checkpoints_(checkpoints),
      targets_(targets), inGraph_(inGraph), gradsProvided_(gradsProvided) {

  if (inGraph_ == InGraph::Yes) {
    if (gradsProvidedFor.size() != gradsProvided.size()) {
      std::ostringstream oss;
      oss << "Objectives with InGraph::Yes must provide "
          << "as many 'gradsProvidedFor' as 'gradsProvided'. ";
      oss << "But " << gradsProvidedFor.size()
          << " tensors were provided for 'gradsProvidedFor': "
          << gradsProvidedFor << " and " << gradsProvided.size()
          << " tensors were provided for 'gradsProvided': " << gradsProvided
          << ". ";
      throw error(oss.str());
    }
  } else {
    if (gradsProvided.size() != 0) {
      std::ostringstream oss;
      oss << "Objectives with InGraph::No must not "
          << "provide any 'gradsProvided' tensors. ";
      throw error(oss.str());
    }
  }

  if (poprithms::util::unisorted(checkpoints).size() != checkpoints.size()) {
    throw error("checkpoints_ not all unique in Objective");
  }
}

const TensorIds &Objective::gradsProvided() const {
  if (inGraph_ == InGraph::No) {
    throw error("Invalid call to gradsProvided() when InGraph::No");
  }
  return gradsProvided_;
}

void Objective::append(std::ostream &ost) const {
  ost << "Objective("
      << "gradsProvidedFor_=" << gradsProvidedFor_
      << ", checkpoints_=" << checkpoints_ << ", targets_=" << targets_
      << ')';
}

std::string Objective::str() const {
  std::ostringstream oss;
  append(oss);
  return oss.str();
}

std::ostream &operator<<(std::ostream &ost, const Objective &g) {
  g.append(ost);
  return ost;
}

Objective Objective::inGraph(const TensorIds &gradsProvidedFor,
                             const TensorIds &checkpoints,
                             const TensorIds &targets,
                             const TensorIds &gradsProvided) {
  return Objective(
      gradsProvidedFor, checkpoints, targets, InGraph::Yes, gradsProvided);
}

Objective Objective::outOfGraph(const TensorIds &gradsProvidedFor,
                                const TensorIds &checkpoints,
                                const TensorIds &targets) {
  return Objective(gradsProvidedFor, checkpoints, targets, InGraph::No, {});
}
} // namespace guide
} // namespace autodiff
} // namespace poprithms
