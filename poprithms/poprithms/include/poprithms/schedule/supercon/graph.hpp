// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SUPER_GRAPH
#define POPRITHMS_SCHEDULE_SUPER_GRAPH

#include <array>
#include <initializer_list>
#include <map>
#include <set>
#include <vector>

namespace poprithms {
namespace schedule {
namespace supercon {

using NodeId = uint64_t;
using Edges  = std::vector<std::vector<NodeId>>;

/**
 * A class to represent a Couple constraint.
 *
 * A Couple constraint is a tuple of four NodeIds, [a,b,c,d], indicating
 * that (a is scheduled before b) if and only if (c is scheduled before d).
 * Valid Couples satisfy the following conditions:
 *
 * * (a != b)             - Op can't be scheduled before itself.
 * * (c != d)             - Op can't be scheduled before itself.
 * * (a != c) OR (b != d) - a,b can't be the same as c,d.
 * * (a != d) OR (b != c) - a,b can't be the reverse of c,d
 *
 * Note that Couples [a,b,c,d], [b,a,d,c], [c,d,a,b] and [d,c,b,a] express
 * equivalent constraints but, e.g., [a,b,d,c] is distinct. Under the hood,
 * we canonically represent Couple constraints by choosing an ordering
 * [a,b,c,d] for which the following conditions hold:
 *
 * * (a == min(a,b))
 * * (min(a,b) < min(c,d)) OR (min(a,b) == min(c,d) AND max(a,b) <= max(c,d))
 *
 * When restricted to Couples that meet the validity constraints above there
 * is always only one ordering that meets both canonicity constraints.
 * */
class Couple {
public:
  /**
   * Custom constructor from non-canonical array of NodeIds.
   * */
  Couple(const std::array<NodeId, 4> &rhs);

  /**
   * Index operator.
   * */
  const NodeId &operator[](std::array<NodeId, 4>::size_type index) const;

  /**
   * Less operator.
   * */
  bool operator<(const Couple &rhs) const;

  /**
   * Equality operator.
   * */
  bool operator==(const Couple &rhs) const;

private:
  // Helper function to canonicalise a std::array<NodeId, 4>.
  static std::array<NodeId, 4>
  canonicalize(const std::array<NodeId, 4> &couple);

  // The canonical array.
  std::array<NodeId, 4> value;
};

/**
 * Stream operator.
 * */
std::ostream &operator<<(std::ostream &out, const Couple &couple);

using Couples = std::vector<Couple>;

/**
 * A minimal graph representation for first-in last-out (Filo) Kahn's
 * algorithm with super constraints.
 *
 * Input arguments:
 *
 * 1) edges :
 * ----------
 * the standard topological constraints of a DAG, that is,
 * b \in edges[a] implies that b appears before a in the schedule.
 *
 * 2) couples :
 * ------------
 * the constraint here is that for all v \in couples,
 *   v[0] before v[1] if and only if v[2] before v[3].
 *
 * As an example, suppose the Graph is
 *
 *    A   E
 *   /|   |\.
 *  B C   F G
 *   \|   |/
 *    D   H
 *
 * and that [B,C,F,G] \in couples.
 *
 * The only valid schedules with this Couple are:
 * ABCDEFGH
 * EFGHABCD
 * ACBDEGFH
 * EGFHACBD .
 *
 * In other words, valid schedules have (B before C) == (F before G).
 *
 * 3) bins :
 * ---------
 * coming soon, see TODO(T19634)
 * */
class Graph {
public:
  /**
   * Construct an empty graph.
   * */
  Graph();

  /**
   * Construct a graph from parameters.
   * \param edges The topological constraints to use in the form of a vector
   *              of vector of NodeIds. The i-th entry contains a list of
   *              all NodeIds describing the forward edges of NodeId i.
   * \param couples The couple constraints to use.
   * */
  Graph(const Edges &edges, const Couples &couples);

  /**
   * Add a forward edge #from -> #to which will ensure #from appears
   * before #to in the schedule. As a side effect of this operation
   * the Graph will grow to comprise all NodeIds numerically smaller
   * or equal to #from and #to, if it does not already.
   * \param from An NodeId representing an operation in a schedule.
   * \param to An NodeId representing an operation in a schedule.
   * */
  void insertEdge(NodeId from, NodeId to);

  /**
   * Method to determine if this Graph contains forward edge
   * #from -> #to.
   * \param from An NodeId representing an operation in a schedule.
   * \param b An NodeId representing an operation in a schedule.
   * \returns True if and only graph contains edge #from -> #to.
   * */
  bool containsEdge(NodeId from, NodeId b) const;

  /**
   * Insert a couple constraint [a,b,c,d] which will ensure that
   * a appears before b in the schedule if and only if c appears
   * before b.
   * \param couple A couple constraint.
   */
  void insertCouple(const Couple &couple);

  /**
   * Method to determine if Graph contains a Couple constraint.
   * This check works on canonical Couple objects.
   * \param couple A couple constraint.
   * \returns True if and only if graph contains couple.
   * */
  bool containsCouple(const Couple &couple) const;

  /**
   * Find a schedule for the given edge and couple constraints.
   * Ths method's implementation is based on Kahn's algorithm.
   * \returns A vector representing the order of ops.
   * */
  std::vector<NodeId> getFiloSchedule() const;

private:
  // Helper function to grow graph to contain NodeId a, if needed.
  void grow(NodeId a);
  // Helper function that returns the number of ops.
  uint64_t nOps() const;

  // The forward edges in the Graph
  Edges edges;

  // Couple constraints in the Graph.
  Couples couples;
};

std::vector<uint64_t> getFiloSchedule(const Edges &forwardEdges,
                                      const Couples &couples
                                      /* bins: (T19634) */
);

} // namespace supercon
} // namespace schedule
} // namespace poprithms

#endif
