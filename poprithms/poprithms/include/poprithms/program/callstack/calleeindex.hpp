// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_CALLSTACK_CALLEEINDEX_HPP
#define POPRITHMS_PROGRAM_CALLSTACK_CALLEEINDEX_HPP

#include <poprithms/util/typedinteger.hpp>

namespace poprithms {
namespace program {
namespace callstack {

/**
 * The id of a sub-graph within an op with callees. For example, the
 * unique callee graph in a call op has id CalleeIndex(0). The callee graphs
 * in a switch op are CalleeIndex(0)...CalleeIndex(n_cases -1).
 *
 * Is it necessary to identify callees by index, why can't they just be
 * identified by a global sub-graph (program) id? Because an op might use a
 * single sub-graph in mulitple cases, with only the input/output tensors
 * changing.
 * */
using CalleeIndex   = poprithms::util::TypedInteger<'C', uint32_t>;
using CalleeIndices = std::vector<CalleeIndex>;

} // namespace callstack
} // namespace program
} // namespace poprithms

#endif
