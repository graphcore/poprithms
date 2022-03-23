// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <autodiff/autodiff/error.hpp>

#include <poprithms/autodiff/core/gradinfo.hpp>

namespace poprithms {
namespace autodiff {
namespace core {

std::vector<GradInfo::CheckpointPair> GradInfo::checkpointPairs() const {
  std::vector<CheckpointPair> cps;
  cps.reserve(objective_.nCheckpoints());
  for (uint64_t i = 0; i < objective_.nCheckpoints(); ++i) {
    cps.push_back({objective_.checkpoint(i), summary_.checkpointsIn()[i]});
  }
  return cps;
}

std::vector<GradInfo::GradInPair> GradInfo::gradInPairs() const {
  std::vector<GradInPair> gps;
  gps.reserve(objective_.nInGrads());
  for (uint64_t i = 0; i < objective_.nInGrads(); ++i) {
    GradInPair p{objective_.gradProvidedFor(i), summary_.gradsIn()[i]};
    gps.push_back(p);
  }
  return gps;
}

namespace {
// get the index of #a in #b.
uint64_t index(const TensorId &a, const TensorIds &b) {
  uint64_t i{0};
  while (i < b.size() && b[i] != a) {
    ++i;
  }
  if (i == b.size()) {
    std::ostringstream oss;
    oss << "failed to find the TensorId " << a
        << " in the vector of TensorIds " << b << ". ";
    throw error(oss.str());
  }
  return i;
}
} // namespace

TensorId GradInfo::gradInputInGradGraph(const TensorId &inNonGrad) const {
  return summary_.gradsIn().at(
      index(inNonGrad, objective_.gradsProvidedFor()));
}
TensorId GradInfo::targetInNonGradGraph(const TensorId &inGrad) const {
  return objective_.target(index(inGrad, summary_.targetGrads()));
}

TensorId GradInfo::checkpointInGradGraph(const TensorId &inNonGrad) const {
  return summary_.checkpointsIn().at(
      index(inNonGrad, objective_.checkpoints()));
}

TensorId GradInfo::targetGradInGradGraph(const TensorId &inNonGrad) const {
  return summary_.targetGrads().at(index(inNonGrad, objective_.targets()));
}

TensorId GradInfo::gradInputInNonGradGraph(const TensorId &inGrad) const {
  return objective_.gradProvidedFor(index(inGrad, summary_.gradsIn()));
}

TensorId GradInfo::checkpointInNonGradGraph(const TensorId &inGrad) const {
  return objective_.checkpoint(index(inGrad, summary_.checkpointsIn()));
}

GradInfo GradInfo::outOfGraph(SubGraphId nonGradSubGraphId,
                              SubGraphId gradSubGraphId,
                              const GradInPairs &grads,
                              const CheckpointPairs &checkpoints,
                              const TargetAndGradPairs &targets) {

  // unzip the pairs of tensors to construct an objective and a summary.
  TensorIds g0;
  TensorIds g1;
  for (const auto &p : grads) {
    g0.push_back(p.gradInGradGraph);
    g1.push_back(p.nonGradInNonGradGraph);
  }

  TensorIds c0;
  TensorIds c1;
  for (const auto &p : checkpoints) {
    c0.push_back(p.inGradGraph);
    c1.push_back(p.inNonGradGraph);
  }

  TensorIds t0;
  TensorIds t1;
  for (const auto &p : targets) {
    t0.push_back(p.gradInGradGraph);
    t1.push_back(p.nonGradInNonGradGraph);
  }

  return GradInfo(nonGradSubGraphId,
                  gradSubGraphId,
                  Objective::outOfGraph(g1, c1, t1),
                  Summary(g0, c0, t0));
}

GradInfo::GradInfo(SubGraphId ngGraph,
                   SubGraphId gGraph,
                   const Objective &fwd,
                   const Summary &bwd)
    : nonGradSubGraphId_(ngGraph), gradSubGraphId_(gGraph), objective_(fwd),
      summary_(bwd) {}

} // namespace core
} // namespace autodiff
} // namespace poprithms
