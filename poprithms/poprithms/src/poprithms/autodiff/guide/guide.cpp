// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <autodiff/autodiff/error.hpp>

#include <poprithms/autodiff/guide/graphinfo.hpp>
#include <poprithms/autodiff/guide/guide.hpp>
#include <poprithms/autodiff/guide/objective.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace autodiff {
namespace guide {

namespace {
std::ostream &operator<<(std::ostream &ost, const std::set<TensorId> &t) {
  ost << std::vector<TensorId>(t.cbegin(), t.cend());
  return ost;
}
} // namespace

Guide::Guide(const Objective &g, const GraphInfo &p)
    : generator(g), graphInfo(p), traversals_(g, p) {

  setNonGradsForAutodiff();
  setNonGradsWithGrads();
  setNonGradsToRecompute();
  setOpsToRerun();
  verifyRecomputeOrder(p, g);
}

std::ostream &operator<<(std::ostream &ost, const Guide &d) {
  d.append(ost);
  return ost;
}

std::map<OpId, int64_t> Guide::getFwdEdgeDependencyCount() const {
  std::map<OpId, int64_t> nAwaiting;
  for (const auto &[k, vs] : fwdEdges()) {
    auto found = nAwaiting.find(k);
    if (found == nAwaiting.cend()) {
      nAwaiting.insert({k, 0});
    }
    for (auto v : vs) {
      found = nAwaiting.find(v);
      if (found == nAwaiting.cend()) {
        nAwaiting.insert({k, 1});
      } else {
        found->second += 1;
      }
    }
  }
  return nAwaiting;
}

void Guide::append(std::ostream &ost) const {

  ost << "\n    autodiff traversals:\n" << traversals_;
  ost << "\n    non gradient tensors for autodiff   = "
      << nonGradsForAutodiff_;
  ost << "\n    non gradient tensors with gradients = " << nonGradsWithGrads_;
  ost << "\n    non gradient tensors to recompute   = "
      << nonGradsToRecompute_;
  ost << "\n    ops to recompute=";
  poprithms::util::append(ost, opsToRerun_);
}

void Guide::setNonGradsWithGrads() {

  // tensors being targeted
  for (auto t : generator.targets()) {
    nonGradsWithGrads_.insert(t);
  }

  // the tensors with gradients provided for
  for (auto t : generator.gradsProvidedFor()) {
    nonGradsWithGrads_.insert(t);
  }

  for (auto traversal : opTraversals()) {

    // all tensors on a differentiable path from a targeted tensor to a tensor
    // with a gradient provided.
    const auto inId = graphInfo.inTensorId(traversal);
    const TensorId outId{traversal.opId(), traversal.outIndex()};
    nonGradsWithGrads_.insert(inId);
    nonGradsWithGrads_.insert(outId);

    // Ops which need to backpropagate might have some outputs which are not
    // on a path to an input gradient. These gradients should still exist
    // for the Ops to run (and they should be zero as they aren't on a path to
    // an initial gradient).
    for (auto otherOut : graphInfo.outTensorIds(traversal.opId())) {
      if (graphInfo.gradientPropagates(otherOut)) {
        nonGradsWithGrads_.insert(otherOut);
      }
    }
  }
}

void Guide::setNonGradsForAutodiff() {

  // For each of the Ops which needs to be differentiated, append a minimal
  // set of non-grad-tensors to a global set, so as to ensure that the
  // autodiff can be performed.
  for (auto opId : getOps(opTraversals())) {
    graphInfo.extendAutodiffRequiredTensors(opId, nonGradsForAutodiff_);
  }
}

std::set<OpId> Guide::getOps(const OpTraversals &traversals) {
  std::set<OpId> os;
  for (const auto &t : traversals) {
    os.insert(t.opId());
  }
  return os;
}

void Guide::setNonGradsToRecompute() {

  // Starting with a stack consisting of all non-grad tensors which must be
  // available at some point for computing gradients,
  TensorIds stack_{nonGradsForAutodiff_.cbegin(),
                   nonGradsForAutodiff_.cend()};
  std::set<TensorId> visited = nonGradsForAutodiff_;

  while (!stack_.empty()) {

    // pop from the back of the stack of required tensors, and check if the
    // popped tensor is checkpointed. If it isn't it must be recomputed. The
    // inputs to its creator might also need to be recomputed, place them on
    // the stack. The stack is processed with a depth first (towards
    // creators) search until empty.
    auto required = stack_.back();
    stack_.pop_back();

    if (!generator.isCheckpoint(required)) {
      graphInfo.assertCanBeRerun(required.opId());
      nonGradsToRecompute_.insert(required);
      for (const auto &inId : graphInfo.inTensorIds(required.opId())) {
        if (visited.count(inId) == 0) {
          stack_.push_back(inId);
          visited.insert(inId);
        }
      }
    }
  }
}

/**
 * All the ops to rerun. These are the ops which are producers of at least 1
 * recomputed tensor. They are returned in topologically sorted order.
 * */
void Guide::setOpsToRerun() {
  std::set<OpId> opsToRerun;
  for (const auto &tId : nonGradsToRecompute_) {
    opsToRerun.insert(tId.opId());
  }
  opsToRerun_ = graphInfo.subSchedule(opsToRerun);
}

void Guide::verifyRecomputeOrder(const GraphInfo &m,
                                 const Objective &g_) const {

  // Go through all of the Ops to rerun in their specified order, and
  // check that all their input tensors are available when they will run. This
  // set contains all the tensors currently available (recomputed or
  // checkpointed). It is initially all checkpointed tensors.
  std::set<TensorId> nonGrads;
  for (auto c : g_.checkpoints()) {
    nonGrads.insert(c);
  }

  for (auto opId : opsToRerun_) {
    for (auto inIdInFwd : m.inTensorIds(opId)) {
      if (nonGrads.find(inIdInFwd) == nonGrads.cend()) {
        std::ostringstream oss;
        oss << "Failure in Guide::verifyRecomputeOrder. ";
        oss << "The op: \n";
        m.appendOpInfo(oss, opId);
        oss << " could not be rerun, as the input tensor " << inIdInFwd
            << " is not available with the (determined) recompute schedule: ";
        poprithms::util::append(oss, opsToRerun_);
        oss << ", and (provided) checkpoints " << g_.checkpoints() << ". ";
        throw error(oss.str());
      }
    }

    // insert the recomputed tensors into the set.
    for (uint64_t o = 0; o < m.nOutTensors(opId); ++o) {
      nonGrads.insert({opId, o});
    }
  }
}
} // namespace guide

} // namespace autodiff
} // namespace poprithms
