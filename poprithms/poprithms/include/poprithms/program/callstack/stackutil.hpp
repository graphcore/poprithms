// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_CALLSTACK_STACKUTIL_HPP
#define POPRITHMS_PROGRAM_CALLSTACK_STACKUTIL_HPP

#include <poprithms/program/callstack/stacktensorid.hpp>

namespace poprithms {
namespace program {
namespace callstack {

class StackUtil {
public:
  /**
   * \return The number of time each TensorId appears in #stIds.
   * */
  static std::map<TensorId, uint64_t> getCounts(const StackTensorIds &stIds);

  /**
   * \return The set of TensorIds which appear in the StackTensorIds #stIds.
   * */
  static std::set<TensorId> tensorIds(const StackTensorIds &stIds);

  /**
   * Create StackTensorIds by combining the TensorIds #tIds  with the
   * CallStack #callStack.
   * */
  static StackTensorIds inScope(const TensorIds &tIds,
                                const CallStack &callStack);

  static StackTensorIds inMainScope(const TensorIds &tIds) {
    return inScope(tIds, CallStack({}));
  }
};

} // namespace callstack
} // namespace program
} // namespace poprithms

#endif
