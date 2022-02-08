// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_CALLSTACK_STACKTENSORID_HPP
#define POPRITHMS_PROGRAM_CALLSTACK_STACKTENSORID_HPP

#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/program/callstack/callstack.hpp>

namespace poprithms {
namespace program {
namespace callstack {

/**
 * A Tensor within a CallStack.
 * */
class StackTensorId {
public:
  StackTensorId(const TensorId &id, const CallStack &st)
      : id_(id), callStack_(st) {}

  const TensorId tId() const { return id_; }
  const CallStack &callStack() const { return callStack_; }

  bool operator!=(const StackTensorId &rhs) const { return t() != rhs.t(); }
  bool operator<(const StackTensorId &rhs) const { return t() < rhs.t(); }
  bool operator<=(const StackTensorId &rhs) const { return t() <= rhs.t(); }
  bool operator==(const StackTensorId &rhs) const { return t() == rhs.t(); }
  bool operator>=(const StackTensorId &rhs) const { return t() >= rhs.t(); }
  bool operator>(const StackTensorId &rhs) const { return t() > rhs.t(); }

  void append(std::ostream &) const;

private:
  TensorId id_;
  CallStack callStack_;
  std::tuple<TensorId, CallStack> t() const { return {id_, callStack_}; }
};

using StackTensorIds = std::vector<StackTensorId>;

std::ostream &operator<<(std::ostream &, const StackTensorId &);
std::ostream &operator<<(std::ostream &, const StackTensorIds &);

} // namespace callstack
} // namespace program
} // namespace poprithms

#endif
