// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <set>

#include <autodiff/autodiff/error.hpp>

#include <poprithms/autodiff/guide/traversals.hpp>
#include <poprithms/common/multiout/traversal.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace autodiff {
namespace guide {

void Traversals::append(std::ostream &ost) const {

  ost << "\n    op traversals=" << opTraversals_;
  poprithms::util::append(ost, opTraversals_);
  ost << "\n    forward edges=";
  for (const auto &[k, vals] : fwdEdges_) {
    ost << "\n        " << k << "->";
    auto opIds = std::vector<OpId>(vals.cbegin(), vals.cend());
    poprithms::util::append(ost, opIds);
  }
}

std::ostream &operator<<(std::ostream &ost, const Traversals &ats) {
  ats.append(ost);
  return ost;
}

Traversals::Traversals(const Objective &g, const GraphInfo &i) {
  using namespace poprithms::common::multiout;

  for (auto tId : g.gradsProvidedFor()) {
    i.assertCanHaveGrad(tId);
  }
  i.assertValidPaths(g.targets(), g.gradsProvidedFor());

  setTraversals(g, i);
  setFwdEdges(i);
}

void Traversals::setTraversals(const Objective &generator,
                               const GraphInfo &graphInfo) {

  auto dp = [&graphInfo](const OpTraversal &ot) {
    return graphInfo.gradientPropagates(ot);
  };

  auto fromTarget = depthFirstForward(graphInfo, generator.targets(), dp);

  auto toInGrads =
      depthFirstBackward(graphInfo, generator.gradsProvidedFor(), dp);

  // opTraversals_ : lie on path from targets and to input gradients.
  std::set_intersection(fromTarget.cbegin(),
                        fromTarget.cend(),
                        toInGrads.cbegin(),
                        toInGrads.cend(),
                        std::back_inserter(opTraversals_));

  // We now set inIndicesTraversed_ and outIndicesTraversed_. The same pattern
  // is used for constructing both maps, so we use lambda functions to avoid
  // code duplication.

  // the input value index is either an InIndex or an OutIndex, and vals is a
  // vector of index's type.
  auto registerTraversedIndex = [](auto &vals, const auto index) {
    if (std::find(vals.cbegin(), vals.cend(), index) == vals.cend()) {
      vals.push_back(index);
    } else {
      // we've already seen a traversal which enters/leaves at nxt.
    }
  };

  auto registerTraversal =
      [&registerTraversedIndex](OpId opId, const auto index, auto &m) {
        auto found = m.find(opId);
        if (found == m.cend()) {
          m.insert({opId, {index}});
        } else {
          registerTraversedIndex(found->second, index);
        }
      };

  std::set<OpId> allOpsTraversed;
  for (const auto &traversal : opTraversals_) {
    registerTraversal(
        traversal.opId(), traversal.inIndex(), inIndicesTraversed_);
    registerTraversal(
        traversal.opId(), traversal.outIndex(), outIndicesTraversed_);
    allOpsTraversed.insert(traversal.opId());
  }

  auto sortVals = [](auto &m) {
    for (auto &[k, v] : m) {
      (void)k;
      std::sort(v.begin(), v.end());
    }
  };
  sortVals(inIndicesTraversed_);
  sortVals(outIndicesTraversed_);

  traversed_ = OpIds(allOpsTraversed.cbegin(), allOpsTraversed.cend());
}

void Traversals::setFwdEdges(const GraphInfo &graphInfo) {
  for (const auto &traversal : opTraversals_) {

    // producer of the input to the traversal, in the non-grad graph. The
    // 'constraint' we insert is traversal.opId() -> out_.
    const auto out_ = graphInfo.inTensorId(traversal).opId();
    auto found      = fwdEdges_.find(traversal.opId());
    if (found == fwdEdges_.cend()) {
      fwdEdges_.insert({traversal.opId(), {out_}});
    } else {
      found->second.insert(out_);
    }
  }
}

InIndices Traversals::inIndicesTraversed(OpId opId) const {
  auto found = inIndicesTraversed_.find(opId);
  if (found == inIndicesTraversed_.cend()) {
    return {};
  }
  return found->second;
}

OutIndices Traversals::outIndicesTraversed(OpId opId) const {
  auto found = outIndicesTraversed_.find(opId);
  if (found == outIndicesTraversed_.cend()) {
    return {};
  }
  return found->second;
}

} // namespace guide
} // namespace autodiff
} // namespace poprithms
