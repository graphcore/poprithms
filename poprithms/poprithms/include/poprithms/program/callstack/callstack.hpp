// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_CALLSTACK_CALLSTACK_HPP
#define POPRITHMS_PROGRAM_CALLSTACK_CALLSTACK_HPP

#include <map>
#include <ostream>
#include <set>
#include <vector>

#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/program/callstack/calleeindex.hpp>

namespace poprithms {
namespace program {
namespace callstack {

using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OpIds;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using poprithms::common::schedulable::SubGraphId;

/**
 * A triplet representing a op's call into a sub-graph, consisting of:
 *
 * 1) an op, which has one or several callee graphs.
 *
 * 2) one of the op's callee graph's ids.
 *
 * 3) the index of the callee graph within the op. For an op with just 1
 *    callee graph (such as a call op, or a repeat op) this index is always 0.
 *    An example of an op with multiple indices is a switch op, where each
 *    switch case will have its own index and probably its own graph.
 * */
class CallEvent {
private:
  OpId caller_;
  SubGraphId callee_;
  CalleeIndex index_;

public:
  CallEvent(OpId caller, SubGraphId callee, CalleeIndex ci)
      : caller_(caller), callee_(callee), index_(ci) {}

  OpId caller() const { return caller_; }
  SubGraphId callee() const { return callee_; }
  CalleeIndex index() const { return index_; }
  uint64_t index_u64() const { return index_.get(); }

  bool operator!=(const CallEvent &rhs) const { return t() != rhs.t(); }
  bool operator<(const CallEvent &rhs) const { return t() < rhs.t(); }
  bool operator<=(const CallEvent &rhs) const { return t() <= rhs.t(); }
  bool operator==(const CallEvent &rhs) const { return t() == rhs.t(); }
  bool operator>=(const CallEvent &rhs) const { return t() >= rhs.t(); }
  bool operator>(const CallEvent &rhs) const { return t() > rhs.t(); }

  void append(std::ostream &) const;

private:
  std::tuple<OpId, SubGraphId, CalleeIndex> t() const {
    return {caller_, callee_, index_};
  }
};

using CallEvents = std::vector<CallEvent>;

/**
 * A call stack. Currently this does not need to be a standalone class,
 * but this might change in the future.
 * */
using CallStack = std::vector<CallEvent>;

std::ostream &operator<<(std::ostream &, const CallEvent &);
std::ostream &operator<<(std::ostream &, const CallStack &);

} // namespace callstack
} // namespace program
} // namespace poprithms

#endif
