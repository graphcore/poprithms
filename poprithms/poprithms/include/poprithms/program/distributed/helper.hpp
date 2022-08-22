// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_DISTRIBUTED_HELPER_HPP
#define POPRITHMS_PROGRAM_DISTRIBUTED_HELPER_HPP

#include <ostream>
#include <vector>

#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/program/distributed/codelocation.hpp>

namespace poprithms {
namespace program {
namespace distributed {

using poprithms::common::schedulable::SubGraphId;
using poprithms::common::schedulable::SubGraphIds;

using OpId  = poprithms::common::multiout::OpId;
using OpIds = poprithms::common::multiout::OpIds;

/**
 * With poplar, a user specifies which poplar::Programs can be dynamically
 * executed from the host. This is done by a poplar::Engine constructor
 * parameter.
 *
 * This diff considers generalized sequences of ops, where ops can execute
 * either on host or ipu (see the CodeLocation enum). As with poplar, the user
 * specifies which of the generalized sequences can be executed dynamically.
 *
 * The goal of this project is to detemine which ops in a generalized
 * sequence correspond to common poplar sequences, and which poplar programs
 * must be made dynanically executable.
 * */
class Helper {
public:
  /**
   * The generalized sub-graphs which must be dynamically executable.
   * */
  virtual SubGraphIds userCallable() const = 0;

  /**
   * All sub-graphs which might be called by #opId.
   * */
  virtual SubGraphIds callees(OpId) const = 0;

  /**
   * Returns the order in which ops in #sgId should be exected. Ops with
   * CodeLocation::None, which might include for example pure view-changing
   * ops, can be optionally omitted from this schedule.
   * */
  virtual OpIds schedule(SubGraphId sgId) const = 0;

  /**
   * The CodeLocation of the op #opId.
   * */
  virtual CodeLocation codeLocation(OpId) const = 0;

  /**
   * All of the sub-graphs which might be executed as a result of executing a
   * sub-graph in the callable set (userCallable). This traverses through all
   * callee sub-graphs of all ops which have callees.
   * */
  SubGraphIds userReachable() const;

private:
  virtual void noWeakVTables();
};

} // namespace distributed
} // namespace program
} // namespace poprithms

#endif
