// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_ALLOCSIMPLIFIER
#define POPRITHMS_SCHEDULE_SHIFT_ALLOCSIMPLIFIER

#include <poprithms/schedule/shift/graph.hpp>
#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

class AllocSimplifier {
public:
  using TransitiveClosure = transitiveclosure::TransitiveClosure;
  using OpAddresses       = std::vector<OpAddress>;
  using AllocAddresses    = std::vector<AllocAddress>;

  /**
   * If 2 Allocations (A and B) are associated to an identical set of Ops,
   * combine them. More specifically, replace A's weight with the sum of A and
   * B's, and remove B.
   *
   * More generally, if a set of Allocations #allocs, of size greater
   * than 1, has all Allocations associated to identical Ops, then set the
   * weight of the first element of #allocs to be the sum of all Allocations
   * in #allocs, and disconnect all the others.
   *
   * For example, if the Graph #graph has Ops (a,b,c) and Allocations (A, B,
   * C) as follows:
   *
   *   A,C    B     A,C
   *   |      |     |
   *   a  ->  b ->  c
   *
   * then replace A's weight with A's weight plus C's weight, and disconnect C
   * from a and c.
   *
   * */
  static bool combineAllocsWithCommonOps(Graph &graph);

  /**
   * If an Allocation is associated to 1 Op, disconnect it from that Op. This
   * will not change the relative livenesses of schedules of #graph.
   * */
  static bool disconnectAllocsWithOneOp(Graph &graph);

  /**
   * If an Allocation has size 0 and it is connected to some Ops, disconnect
   * it from those Ops. Size 0 Allocations do not contribute to the liveness
   * of a schedule.
   * */
  static bool disconnectAllocsWithZeroWeight(Graph &);

  /**
   * If an Allocation #a is associated to an Op #o, where #o is definitely
   * never the first or last Op associated to #a in any schedule, then #o can
   * be disassocated from #a. Only the first and last Ops associated to an
   * Allocation effect the Allocations contribution to the liveness of a
   * schedule.
   *
   * Example: Consider "the diamond",
   *
   * A      A        A
   * |      |        |
   * a -+---b---+--> c
   *    |       |
   *    +----d--+
   *         |
   *         A
   *
   * A can be removed from b and d, as these 2 Ops are never the first or
   * last Ops associated to A.
   *
   * \param g The Graph which might have Ops disassocated from Allocs.
   *
   * \param tc The TransitiveClosure is used to determine if an Op #o is ever
   *           the first or last to be scheduled, relative to all the Ops
   *           associated to #a.
   * */
  static bool disconnectInbetweenerAllocs(Graph &g,
                                          const TransitiveClosure &tc);

  /**
   * If the duration that an Allocation #a will be live for fixed for all
   * schedules, that is if:
   *    MIN_(all schedules)[
   *        max_schedule_index(ops associated to a) -
   *                        min_schedule_index(ops associated to a)] =
   *    MAX_(all schedules)[
   *        max_schedule_index(ops associated to a) -
   *                        min_schedule_index(ops associated to a)],
   *
   * then #a can be disassociated from all of its Ops, without changing the
   * relative livenesses of the schedules.
   */
  static bool disconnectFixedDurationAllocs(Graph &g,
                                            const TransitiveClosure &tc);

  /**
   * If for some Op #o, there is an Allocation #a which is definitely first
   * for #o, and another Allocation #z which is definitely last for #o, then
   * if #a and #z have the same size, a and z can be merged without changing
   * the difference in liveness between any 2 schedules.
   *
   * Example
   *
   * A      A,B      B,C     C
   * |       |       |       |
   * a ----> b ----> c ----> d
   *
   * A and B can be "merged", as can B and C, to create:
   *
   * A       A       A       A
   * |       |       |       |
   * a ----> b ----> c ----> d
   *
   * */
  static bool connectContiguousAllocs(Graph &g, const TransitiveClosure &tc);
};

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
