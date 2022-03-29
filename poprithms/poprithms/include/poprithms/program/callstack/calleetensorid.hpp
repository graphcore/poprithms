// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_CALLSTACK_CALLEETENSORID_HPP
#define POPRITHMS_PROGRAM_CALLSTACK_CALLEETENSORID_HPP

#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/program/callstack/callstack.hpp>
#include <poprithms/util/valuedtuple.hpp>

namespace poprithms {
namespace program {
namespace callstack {

/**
 * A tensor within a CalleeIndex.
 * */
class CalleeTensorId
    : public util::ValuedTuple<std::tuple<TensorId, CalleeIndex>> {
public:
  CalleeTensorId(const TensorId &id, CalleeIndex ci)
      : ValuedTuple({id, ci}) {}
  TensorId tId() const { return get<0, TensorId>(); }
  CalleeIndex calleeIndex() const { return get<1, CalleeIndex>(); }
  void append(std::ostream &) const;
};

using CalleeTensorIds = std::vector<CalleeTensorId>;

std::ostream &operator<<(std::ostream &, const CalleeTensorId &);
std::ostream &operator<<(std::ostream &, const CalleeTensorIds &);

} // namespace callstack
} // namespace program
} // namespace poprithms

#endif
