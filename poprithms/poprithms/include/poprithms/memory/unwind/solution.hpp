// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_UNWIND_SOLUTION_HPP
#define POPRITHMS_MEMORY_UNWIND_SOLUTION_HPP

#include <map>

#include <poprithms/memory/chain/chain.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/path.hpp>
#include <poprithms/memory/unwind/valuedtensorid.hpp>
#include <poprithms/util/typedinteger.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

using common::multiout::TensorIds;
using nest::DisjointRegions;

/**
 * Algorithms for setting the Paths to all Tensors from Sources or Barriers.
 *
 * Brief descriptions:
 *
 * Algo::Greedy0
 * -------------
 * Repeat until all Tensors have their entire layout set:
 *
 * 1) While there is an unwinable Op which can map a layout forwards or
 *    backwards, it must do this, filling in any Regions in Tensors where
 *    the layout is unknown and leaving already set Regions alone.
 *
 * 2) Choose the most valuable ValuedPair which has a Region which has a
 *    known layout (a path from a Source or a Barrier) in one Tensor, but
 *    not the other. Copy the layout to the Tensor with the unset Region.
 *
 * */
enum class Algo { Greedy0 };

/**
 * A Solution to the unwinding problem for a specific Graph.
 * */
class Solution {

public:
  /** Construct a Solution for the Graph \a g, using the algorithm \a a. */
  Solution(const Graph &g, Algo a = Algo::Greedy0);

  /** Construct a Solution for the Graph \a g, using the algorithm \a a. */
  Solution(Graph &&g, Algo = Algo::Greedy0);

  /**
   * Create a Solution based on a partial solution in the form of a sequence
   * of Paths to Sinks. This can be useful if you want to determine the
   * layouts of internal Tensors (inwardsPaths) or the objective function
   * (getScore) based on a solution obtained by other means.
   * */
  Solution(const Graph &, const Paths &sourcesAndBarriersToSinks);

  /**
   * Compute and return the score, also known as the objective function, of
   * this Solution.
   * */
  double getScore() const;

  /**
   * Return a sequence of Paths from Sources/Barriers to Sinks.
   *
   * The order of the Paths is important. If the Paths are used for unwinding
   * in the provided order to set the layouts of the target application's Sink
   * Tensors (for example input poplar::Tensors), it is guarantees that all
   * required Barrier layouts will be availale when needed.
   */
  const Paths &sourcesAndBarriersToSinks() const;

  /**
   * Access to the layout Paths to Tensor \a id from Sources or Barriers. This
   * method can be used for any Tensor, not only Sink Tensors. This is useful
   * to determine the layout of any internal Tensor, in terms of Sources and
   * Barriers.
   * */
  const Paths &inwardsPaths(const TensorId &id) const;

  /**
   * The Graph which this Solution was constructed for. A Solution stores its
   * own copy of this Graph, and this method returns a method to that copy.
   * */
  const Graph &graph() const { return graph_; }

  void append(std::ostream &) const;

private:
  /** All Paths from \a tId to a Sink. */
  const Paths &pathsBackToSinks(const TensorId &tId) const;

  void append(const Path &p) { sourcesAndBarriersToSinks_.push_back(p); }

  /**
   * Set the Paths to all Tensors from Sources or Barriers, based on the set
   * of Paths to Sinks in \a soln.
   * */
  void setAllPaths(const Paths &sourcesAndBarriersToSinks);

  /**
   * \return true if the Tensor \a tId has a Path to every element from a
   *         Source or a Barrier. Once the Graph is fully constructed, this
   *         should always be true for all Tensors.
   * */
  bool completelyCoveredByPaths(const TensorId &tId) const;

  /**
   * \return true if all the outputs of Op \a opId have Paths to every element
   *         from a Source or a Barrier.
   * */
  bool completelyCoveredByPaths(OpId opId) const;

  /**
   * \return true if all the Tensors in \a tIds have a Path to every element
   *        from a Source or a Barrier.
   * */
  bool completelyCoveredByPaths(const TensorIds &tIds) const;

  /**
   * \return true if all the Tensors in this Graph have a Path to every
   *         element from a Source or a Barrier.
   * */
  bool completelyCoveredByPaths() const;

  /**
   * \return The DisjointRegions of Tensor \a tId which have Paths from
   *         Sources or Barriers.
   * */
  DisjointRegions coveredByPaths(const TensorId &) const;

  // Paths which might have downstream Tensors which can have their Paths set
  Paths pathStack;

  // Paths to all Tensors from Sources and Barriers.
  std::vector<std::vector<Paths>> inwardsPaths_;

  // Paths from all Tensors to Sink Tensors.
  std::vector<std::vector<Paths>> pathsBackToSinks_;

  // All the Paths from Source and Barrier Tensors, to Sink Tensors.
  // They appear in the order required to satisfy dependencies,
  // where Barrier Ops require all input Tensors before their layout
  // can be set.
  Paths sourcesAndBarriersToSinks_;

  void clearInwardsPaths(OpId);
  void insertInwardsPath(const TensorId &, const Path &);

  void clearPathsBackToSinks(OpId);
  void insertPathBackToSink(const TensorId &, const Path &);

  // Clear inwards Paths, clear Paths back to Sinks, clear pathStack.
  void resetAllPathInfo();
  void insertPath(const Path &);

  void setPathsBackToSinks();
  void setPathStackToSources();

  void assertCompletelyCovererdByPaths() const;

  // return the Ids of all Tensors which are extended in the growing process.
  TensorIds processPathStack();
  void processUnwindPath(const Path &unwindPath);

  void setPathsGreedy0();

  void initPaths();

  const Graph graph_;
};

std::ostream & operator<<(std::ostream &, const Solution &);

} // namespace unwind
} // namespace memory
} // namespace poprithms

#endif
