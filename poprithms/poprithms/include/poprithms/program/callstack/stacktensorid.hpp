// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_CALLSTACK_STACKTENSORID_HPP
#define POPRITHMS_PROGRAM_CALLSTACK_STACKTENSORID_HPP

#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/program/callstack/callstack.hpp>
#include <poprithms/util/valuedtuple.hpp>

namespace poprithms {
namespace program {
namespace callstack {

/**
 * A Tensor within a CallStack.
 * */
class StackTensorId
    : public util::ValuedTuple<std::tuple<TensorId, CallStack>> {
public:
  StackTensorId(const TensorId &id, const CallStack &st)
      : ValuedTuple({id, st}) {}
  TensorId tId() const { return get<0, TensorId>(); }
  const CallStack &callStack() const { return get<1, CallStack>(); }
  void append(std::ostream &) const;
};

using StackTensorIds = std::vector<StackTensorId>;
std::ostream &operator<<(std::ostream &, const StackTensorId &);
std::ostream &operator<<(std::ostream &, const StackTensorIds &);

} // namespace callstack
} // namespace program
} // namespace poprithms

#endif
