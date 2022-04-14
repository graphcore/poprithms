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

  /**
   * Create a vector of CalleeTensorIds, with TensorIds from #tIds and all
   * with CalleeIndex #ci.
   * */
  static std::vector<CalleeTensorId> zip(const TensorIds &tIds,
                                         CalleeIndex ci);

  /**
   * Create a vector of CalleeTensorIds, with TensorIds from #tIds tied to the
   * CalleeIndices in #cis.
   * */
  static std::vector<CalleeTensorId> zip(const TensorIds &tIds,
                                         const std::vector<CalleeIndex> &cis);
};

using CalleeTensorIds = std::vector<CalleeTensorId>;

std::ostream &operator<<(std::ostream &, const CalleeTensorId &);
std::ostream &operator<<(std::ostream &, const CalleeTensorIds &);

} // namespace callstack
} // namespace program
} // namespace poprithms

#endif
