// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_SCHEDULER_HPP
#define POPRITHMS_COMMON_COMPUTE_SCHEDULER_HPP

#include <poprithms/common/compute/graph.hpp>

namespace poprithms {
namespace common {
namespace compute {

using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OpIds;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using poprithms::common::schedulable::FwdEdgeMap;

/**
 * Performs certain scheduling tasks for this Graph (compute::Graph) class.
 * Note that there are other scheduling tasks performed by the
 * schedulable::Graph class (a base class of the compute::Graph class), the
 * methods in the class below are specific to the compute::Graph.
 * */
class Scheduler {

public:
  /**
   * \return Any valid schedule based on the FwdEdgeMap created with
   *         #loweringFwdEdgeMap.
   * */
  static OpIds vanillaLoweringSchedule(const Graph &);

  /**
   * If a tensor #t0 in sub-graph  #sg0 is referenced from a tensor #t1 in
   * #sg1, then there is dependency between the sub-graphs sg1 -> sg0. This
   * method returns the sub-graphs scheduled according to these dependencies.
   *
   * If there is a cycle, an error is thrown. For example:
   * <code>
   *  t0 = sg0.variable()  // t0 in sg0
   *  t1 = t0.refTo_(sg1)  // t1 in sg1
   *  t2 = t1.relu()       // t2 in sg1
   *  t3 = t1.refTo_(sg0)  // t3 in sg0
   * </code>
   *
   * has a cycle:
   *  1) sg0 -> sg1 (because t1 in sg1 requires t0 in sg0).
   *  2) sg1 -> sg0 (because t3 in sg0 requires t1 in sg1).
   *
   * (1) and (2) creates a cycle (sg0 -> sg1 -> sg0).
   * */
  static SubGraphIds scheduleByRefs(const Graph &);

  /**
   * \return A valid schedule of all the non-initializing ops in the sub-graph
   *         #sgId.
   * */
  static OpIds vanillaComputeSchedule(const Graph &, SubGraphId sgId);

private:
  /**
   * Edge map of a topologically valid lowering ordering.
   *
   * This method takes into account:
   *
   * (1) Data dependencies.
   *
   * (2) Control dependencies between non-initializing ops. That is, ops which
   *     DO execute code.
   *
   * (3) Control dependencies between initializing ops. These ops do not
   *     execute code and therefore can be scheduled anywhere. Constraints
   *     between initializing ops can imply constraints between
   *     non-initializing ops, and so these must be correctly transferred. \sa
   *     Graph::isConstraintPhobic.
   *
   *
   * (4) Tensors which reference tensors in other graphs. \sa the RefFrom op.
   *
   * (5) Ops with callees (all ops in callees must be scheduled before the
   *     calling op).
   * */
  static FwdEdgeMap loweringFwdEdgeMap(const Graph &);
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
