// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <schedule/shift/allocsimplifier.hpp>
#include <schedule/shift/error.hpp>
#include <schedule/shift/transitiveclosureconstrainer.hpp>
#include <schedule/shift/transitiveclosureoptimizer.hpp>
#include <schedule/shift/updatefromfirstfinal.hpp>

#include <poprithms/schedule/scc/scc.hpp>
#include <poprithms/schedule/shift/logging.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

void TransitiveClosureOptimizer::apply(
    const TransitiveClosureOptimizations &tcos,
    Graph &g,
    TimeLogger &timeLogger_) {
  TransitiveClosureOptimizer(tcos, g, timeLogger_);
}

void TransitiveClosureOptimizer::initializeTransitiveClosure() {

  const auto stopwatch =
      timeLogger().scopedStopwatch("initializeTransitiveClosure");

  // Before initializing the transitive closure, while we still have the
  // context strings for the Ops in the Graph, confirm that the Graph is
  // schedulable (contains no cycles) and provide a clear error message if it
  // is not.
  if (!ScheduledGraph::isSchedulable(getGraph().getFwdEdges_u64())) {
    const auto &g = getGraph();
    std::vector<std::string> dbs;
    dbs.reserve(g.nOps());
    for (uint64_t i = 0; i < g.nOps(); ++i) {
      dbs.push_back(g.getOp(i).getDebugString());
    }
    std::ostringstream oss;
    oss << "Not all Ops were scheduled while "
        << "initializing the transitive closure, "
        << "there is a cycle in the Graph."
        << " The non-singleton strongly connected components, "
        << "in topological order, are:"
        << scc::getSummary(
               g.getFwdEdges_u64(), dbs, scc::IncludeSingletons::No);

    throw error(oss.str());
  }

  transitiveClosure =
      transitiveclosure::TransitiveClosure(graph.getForwardEdges());
  finalizeTransitiveClosure();
}

void TransitiveClosureOptimizer::removeRedundantEdges() {

  const auto stopwatch = timeLogger().scopedStopwatch("removeRedundantEdges");

  const auto fwdEdges = graph.getForwardEdges();

  const auto redundants = transitiveClosure.getFlattenedRedundants(fwdEdges);
  log().debug("Removing " + std::to_string(redundants.size()) +
              " redundant TransitiveClosure edges/constraints.");
  for (const auto x : redundants) {
    graph.removeConstraint(std::get<0>(x), std::get<1>(x));
  }
}

void TransitiveClosureOptimizer::reinitializeTransitiveClosure() {
  const auto stopwatch =
      timeLogger().scopedStopwatch("reinitializeTransitiveClosure");
  transitiveClosure.bidirectionalPropagate(graph.getForwardEdges());
  finalizeTransitiveClosure();
}

void TransitiveClosureOptimizer::finalizeTransitiveClosure() {

  const auto stopwatch =
      timeLogger().scopedStopwatch("finalizeTransitiveClosure");

  removeRedundantEdges();

  auto zero        = AllocWeight::zero();
  lowerBoundChange = std::vector<AllocWeight>(nOps(), zero);
  upperBoundChange = std::vector<AllocWeight>(nOps(), zero);

  // initializing lowerBoundChange and upperBoundChange
  log().debug("Initializing lowerBoundChange and upperBoundChange.");
  for (const auto &alloc : graph.getAllocs()) {
    auto relativePositions =
        transitiveClosure.getExtremumStatuses(alloc.getOps());

    // Logic check:
    if (relativePositions.size() != alloc.getOps().size()) {
      std::ostringstream oss;
      oss << "There were " << alloc.getOps().size()
          << " passed into the function getExtremumStatuses, but "
          << relativePositions.size()
          << " values were returned. There should be value entry returned "
          << "for every Op. ";
      throw error(oss.str());
    }

    for (uint64_t opIndex = 0; opIndex < alloc.nOps(); ++opIndex) {
      auto opId = alloc.getOps()[opIndex];
      updateFromFirstFinal(lowerBoundChange[opId],
                           upperBoundChange[opId],
                           alloc.getWeight(),
                           relativePositions[opIndex]);
    }
  }
}

void TransitiveClosureOptimizer::updateTransitiveClosure(
    const std::vector<std::vector<OpAddress>> &edges) {
  const auto stopwatch =
      timeLogger().scopedStopwatch("updateTransitiveClosure");

  if (log().shouldLog(logging::Level::Debug)) {
    std::ostringstream oss;
    oss << "Updating TransitiveClosure with "
        << std::accumulate(
               edges.cbegin(),
               edges.cend(),
               0,
               [](size_t a, const auto &x) { return a + x.size(); })
        << " new constraints. ";
    log().debug(oss.str());
  }

  transitiveClosure.update(edges);
  finalizeTransitiveClosure();
}

void TransitiveClosureOptimizer::applyTransitiveClosureOptimizations(
    const TransitiveClosureOptimizations &tco) {

  const auto sw0 =
      timeLogger().scopedStopwatch("applyTransitiveClosureOptimizations");

  {
    std::ostringstream oss;
    oss << "Applying TransitiveClosureOptimizations, \n" << tco;
    log().debug(oss.str());
  }

  // All of the TransitiveClosureOptims to run.
  const auto allToRun = tco.enabled();
  if (allToRun.empty()) {
    return;
  }

  std::vector<TransitiveClosureOptim> roundStack;
  std::vector<TransitiveClosureOptim> nxtRoundStack = allToRun;
  bool wasChangeInLastRound{true};
  int iteration{0};

  std::vector<std::vector<OpAddress>> prevGraphEdges;
  while (wasChangeInLastRound && iteration < tco.maxIterations()) {

    const auto iterStr = "iteration = " + std::to_string(iteration);

    if (iteration == 0) {
      log().debug("Initializing TransitiveClosure (round 0)," + iterStr);
      initializeTransitiveClosure();
    } else {
      const auto dff = graph.constraintDiff(prevGraphEdges);
      // As Updating a TransitiveClosure takes significantly more time for a
      // large number of edges, we prefer to re-initialize when the number of
      // edges is "large";
      const int nLarge = graph.nOps_i32() / 10;
      if (std::accumulate(
              dff.cbegin(), dff.cend(), 0, [](size_t x, const auto &y) {
                return x + y.size();
              }) < nLarge) {

        log().debug("Updating TransitiveClosure, " + iterStr);
        updateTransitiveClosure(dff);
      } else {
        log().debug("Re-initializing TransitiveClosure,  " + iterStr);
        reinitializeTransitiveClosure();
      }
    }

    // The TransitiveClosureOptims to run in this round:
    roundStack = nxtRoundStack;

    {
      std::ostringstream oss;
      oss << "Will run " << roundStack << " in this round. ";
      log().debug(oss.str());
    }

    // All of the TransitiveClosureOptims in 'roundStack' which cause a change
    // will be run again in the next round.
    nxtRoundStack = {};

    auto apply = [this](TransitiveClosureOptim optim) {
      auto getTCConstrainer = [this]() {
        return TransitiveClosureConstrainer(
            graph, transitiveClosure, lowerBoundChange, upperBoundChange);
      };

      const auto name = TransitiveClosureOptimizations::str(optim);
      log().debug("Applying TCO " + name);
      const auto tcoStopwatch =
          timeLogger().scopedStopwatch("Applying TCO " + name);

      switch (optim) {

      case TransitiveClosureOptim::DisconnectAllocsWithZeroWeight: {
        return AllocSimplifier::disconnectAllocsWithZeroWeight(graph);
      }

      case TransitiveClosureOptim::ConnectContiguousAllocs: {
        return AllocSimplifier::connectContiguousAllocs(graph,
                                                        transitiveClosure);
      }

      case TransitiveClosureOptim::DisconnectAllocsWithOneOp: {
        return AllocSimplifier::disconnectAllocsWithOneOp(graph);
      }

      case TransitiveClosureOptim::DisconnectInbetweenerAllocs: {
        return AllocSimplifier::disconnectInbetweenerAllocs(
            graph, transitiveClosure);
      }

      case TransitiveClosureOptim::DisconnectFixedDurationAllocs: {

        // TODO(T44615) create follow up task to investigate slowdown.
        /*
                 return AllocSimplifier::disconnectFixedDurationAllocs(
                     graph, transitiveClosure);
         */

        return false;
      }

      case TransitiveClosureOptim::SlideLinks: {
        return getTCConstrainer().slideLinks();
      }

      case TransitiveClosureOptim::LinkTightDrops: {
        return getTCConstrainer().linkTightDrops();
      }
      case TransitiveClosureOptim::LinkCloseTightPairs: {
        return getTCConstrainer().linkCloseTightPairs();
      }
      case TransitiveClosureOptim::ConstrainWeightSeparatedGroups: {
        return getTCConstrainer().constrainWeightSeparatedGroups();
      }
      case TransitiveClosureOptim::ConstrainParallelChains: {
        return getTCConstrainer().constrainParallelChains();
      }

      case TransitiveClosureOptim::CombineAllocsWithCommonOps: {
        return AllocSimplifier::combineAllocsWithCommonOps(graph);
      }

      case TransitiveClosureOptim::N: {
        throw error("N is not an optimizing TransitiveClosureOptim ");
      }
      }

      throw error("Unrecognized TransitiveClosureOptim");
    };

    log().debug("Storing Graph edges, to detect changes in next iteration");
    prevGraphEdges = graph.getForwardEdges();

    for (auto optim : roundStack) {
      auto wasChange = apply(optim);
      if (wasChange) {
        nxtRoundStack.push_back(optim);
      }
    }

    // There were no changes in this round. If all of the optimizations were
    // tried in this round, then we should stop optimizing. If not all there
    // tried, then we should have a round where all are tried.
    if (nxtRoundStack.empty()) {
      if (roundStack.size() == allToRun.size()) {
        wasChangeInLastRound = false;
      } else {
        nxtRoundStack = allToRun;
      }
    }
    // If only one optimization pass resulted in a change, we assume running
    // it again by itself will have no effect and rather run all.
    if (nxtRoundStack.size() == 1) {
      nxtRoundStack = allToRun;
    }

    ++iteration;
  }
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
