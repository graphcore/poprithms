// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <schedule/shift/transitiveclosureconstrainer.hpp>
#include <schedule/shift/updatefromfirstfinal.hpp>

#include <poprithms/schedule/shift/filteredschedule.hpp>
#include <poprithms/schedule/shift/logging.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

bool TransitiveClosureConstrainer::constrainParallelChains() const {

  std::vector<std::array<OpAddress, 2>> newConstraints;
  for (OpAddress a = 0; a < graph.nOps(); ++a) {
    auto identicalIns = graph.getIdenticalIns(a);
    if (identicalIns.size() <= 1) {
      continue;
    }
    const auto aChain = graph.tightChainFrom(a);
    const auto aEnd   = aChain.back();
    for (auto b : identicalIns) {
      if (b == a) {
        continue;
      }
      auto bChain       = graph.tightChainFrom(b);
      auto bEnd         = bChain.back();
      const auto &aOuts = graph.getOp(aEnd).getOuts();
      const auto &bOuts = graph.getOp(bEnd).getOuts();
      if (!(aOuts == bOuts && (aChain.size() >= bChain.size()))) {
        continue;
      }

      auto chainLength = bChain.size();
      bool canInsertConstraints{true};

      auto runningUpp = AllocWeight::zero();
      auto runningLow = AllocWeight::zero();

      for (uint64_t i = 0; i < chainLength; ++i) {

        auto uppA = upperBoundChange[aChain[i]];
        auto lowB = lowerBoundChange[bChain[i]];

        for (auto allocAddress : graph.getOp(bChain[i]).getAllocs()) {

          // Determine if a shared alloc can be removed:
          bool canRemove = true;
          for (uint64_t j = 0; j < chainLength; ++j) {
            if (graph.getOp(aChain[i]).hasAlloc(allocAddress) !=
                graph.getOp(bChain[i]).hasAlloc(allocAddress)) {
              canRemove = false;
            }
          }
          if (!canRemove) {
            continue;
          }

          // Remove shared: alloc contribution
          const auto &alloc = graph.getAlloc(allocAddress);
          const auto &all   = alloc.getOps();
          auto negW         = -1 * alloc.getWeight();

          AllocWeight dummy = AllocWeight::zero();
          {

            const auto relPoss =
                transitiveClosure.getExtremumStatus(aChain[i], all);
            updateFromFirstFinal(dummy, uppA, negW, relPoss);
          }

          {

            const auto relPoss =
                transitiveClosure.getExtremumStatus(bChain[i], all);
            updateFromFirstFinal(lowB, dummy, negW, relPoss);
          }
        }

        runningUpp += uppA;
        runningLow += lowB;

        if (runningUpp < runningLow ||
            (runningUpp == runningLow && aChain[i] < bChain[i])) {
        } else {
          canInsertConstraints = false;
          break;
        }
      }

      if (canInsertConstraints) {
        for (uint64_t i = 0; i < chainLength; ++i) {
          if (!graph.getOp(aChain[i]).hasOut(bChain[i])) {
            newConstraints.push_back({aChain[i], bChain[i]});
          }
        }
      }
    }
  }
  for (auto constraint : newConstraints) {
    auto from = std::get<0>(constraint);
    auto to   = std::get<1>(constraint);
    graph.insertConstraint(from, to);
  }

  log().debug(
      std::to_string(newConstraints.size()) +
      " new constraints inserted in graph::constrainParallelChains()");

  return !newConstraints.empty();
}

bool TransitiveClosureConstrainer::slideLinks() const {

  bool wasChange{false};
  auto linkChains = graph.getLinkChains();
  for (const auto &chain : linkChains) {
    for (uint64_t i = 0; i < chain.size(); ++i) {
      auto id = chain[i];

      if (i != chain.size() - 1) {
        const auto outs = graph.getOp(id).getOuts();
        for (const auto outId : outs) {
          if (graph.getOp(id).getForwardLink() != outId) {
            graph.removeConstraint(id, outId);
            graph.insertConstraint(chain.back(), outId);
            wasChange |= true;
          }
        }
      }
      if (i != 0) {
        const auto ins = graph.getOp(id).getIns();
        for (const auto inId : ins) {
          if (graph.getOp(id).getBackwardLink() != inId) {
            graph.removeConstraint(inId, id);
            graph.insertConstraint(inId, chain[0]);
            wasChange |= true;
          }
        }
      }
    }
  }

  return wasChange;
}

bool TransitiveClosureConstrainer::linkCloseTightPairs() const {

  std::vector<std::array<OpAddress, 2>> newLinks;

  for (const auto tightPair : graph.getTightPairs()) {
    auto before = std::get<0>(tightPair);
    auto after  = std::get<1>(tightPair);
    if (graph.getOp(before).hasForwardLink()) {
      continue;
    }

    auto L = std::min(lowerBoundChange[before], lowerBoundChange[after]);
    auto U = std::max(upperBoundChange[before], upperBoundChange[after]);

    auto getCanTie = [this, L, U](OpAddress opId) {
      using namespace transitiveclosure;
      for (uint64_t bitSetIndex = 0;
           bitSetIndex < transitiveClosure.getNBitSets(opId);
           ++bitSetIndex) {

        // A step to accelerate the optimization:
        bool unconstrainedInBitSet =
            transitiveClosure.unconstrainedWithAtLeastOne(opId, bitSetIndex);

        if (unconstrainedInBitSet) {
          for (uint64_t shift = 0; shift < BitSetSize; ++shift) {
            auto id = bitSetIndex * BitSetSize + shift;

            if (id != opId && id < graph.nOps() &&
                transitiveClosure.unconstrainedInBothDirections(id, opId)) {
              //      L     U
              //  ....xxxxxxx..  -- a
              //  ..xxxxx......  -- b
              //    l   u
              //  ==> intersection if L < u && l < U
              const auto u = upperBoundChange[id];
              const auto l = lowerBoundChange[id];

              if (L < u && l < U) {
                return false;
              }
            }
          }
        }
      }
      return true;
    };

    bool canTie = getCanTie(before);

    if (canTie) {
      if (!graph.getOp(before).hasForwardLink()) {
        newLinks.push_back(tightPair);
      }
    }
  }

  for (auto link : newLinks) {
    graph.insertLink(std::get<0>(link), std::get<1>(link));
  }
  log().debug(std::to_string(newLinks.size()) +
              " new links inserted in Graph::linkCloseTightPairs()");
  return !newLinks.empty();
}

void TransitiveClosureConstrainer::processWeightSeparatedIdenticalIns(
    const std::vector<OpAddress> &identicalIns,
    std::vector<std::array<OpAddress, 2>> &newConstraints) const {

  // for (a,b) can we insert a'->b for any a' which are post a?
  for (auto a : identicalIns) {
    for (auto b : identicalIns) {
      if (upperBoundChange[a] <= lowerBoundChange[b] && a != b) {

        // Here we do a depth first search, starting at b, stopping when we
        // reach an Op with is unconstrained with respect
        // to t a.
        //
        // The Ops found end up in this vector:
        std::vector<OpAddress> postBs;
        std::vector<OpAddress> toProcess{b};
        std::vector<OpAddress> seen{b};
        while (!toProcess.empty()) {
          const auto nxt = toProcess.back();
          toProcess.pop_back();
          if (!transitiveClosure.constrained(a, nxt)) {
            postBs.push_back(nxt);
            for (auto out : graph.getOp(nxt).getOuts()) {
              if (std::find(seen.cbegin(), seen.cend(), out) == seen.cend()) {
                seen.push_back(out);
                toProcess.push_back(out);
              }
            }
          }
        }

        auto lb = lowerBoundChange[b];
        for (auto postB : postBs) {
          lb = std::min(lb, lowerBoundChange[postB]);
        }

        if (upperBoundChange[a] <= lb) {

          auto nPostBoth  = transitiveClosure.nPostPost(a, b);
          auto candidates = getFilteredSchedule(
              graph, a, [this, lb, b, nPostBoth](OpAddress x) {
                return upperBoundChange[x] <= lb &&
                       (transitiveClosure.nPostPost(b, x) == nPostBoth);
              });

          if (a < b || std::any_of(candidates.cbegin(),
                                   candidates.cend(),
                                   [lb, this](OpAddress postA) {
                                     return upperBoundChange[postA] < lb;
                                   })) {
            for (auto aPrime : candidates) {
              newConstraints.push_back({aPrime, b});
            }
          }
        }
      }
    }
  }
}

bool TransitiveClosureConstrainer::constrainWeightSeparatedGroups() const {

  std::vector<bool> processed(graph.nOps(), false);

  std::vector<std::array<OpAddress, 2>> newConstraints;
  for (OpAddress add0 = 0; add0 < graph.nOps(); ++add0) {
    if (processed[add0]) {
      continue;
    }
    auto identicalIns = graph.getIdenticalIns(add0);
    for (auto id0 : identicalIns) {
      processed[id0] = true;
    }

    if (identicalIns.size() < 2) {
      continue;
    }

    processWeightSeparatedIdenticalIns(identicalIns, newConstraints);
  }

  for (auto constraint : newConstraints) {
    auto from = std::get<0>(constraint);
    auto to   = std::get<1>(constraint);
    graph.insertConstraint(from, to);
  }

  log().debug(
      std::to_string(newConstraints.size()) +
      " new constraints inserted in graph::constrainWeightSeparatedGroups()");

  return !newConstraints.empty();
}

bool TransitiveClosureConstrainer::linkTightDrops() const {

  std::vector<std::array<OpAddress, 2>> newLinks;
  for (const auto tightPair : graph.getTightPairs()) {
    OpAddress before = std::get<0>(tightPair);
    OpAddress after  = std::get<1>(tightPair);
    if (upperBoundChange[after] <= lowerBoundChange[before]) {
      if (!graph.getOp(before).hasForwardLink() &&
          !graph.getOp(after).hasBackwardLink()) {
        newLinks.push_back(tightPair);
      }
    }
  }
  for (auto link : newLinks) {
    graph.insertLink(std::get<0>(link), std::get<1>(link));
  }
  log().debug(std::to_string(newLinks.size()) +
              " new links inserted in Graph::linkTightDrops()");
  return !newLinks.empty();
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
