// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_UNWIND_USINGS_HPP
#define POPRITHMS_MEMORY_UNWIND_USINGS_HPP

#include <ostream>
#include <utility>
#include <vector>

#include <poprithms/util/typedinteger.hpp>
#include <poprithms/util/typedvector.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

/**
 * SubGraphIds defines the scopes of Ops and Tensors. Ops in the "main"
 * computation graph, and Ops in the various computation subgraphs in calls,
 * loops, and condititionals, should have distinct SubGraphIds.
 * */
using SubGraphId  = poprithms::util::TypedInteger<'g', uint32_t>;
using SubGraphIds = std::vector<SubGraphId>;
std::ostream &operator<<(std::ostream &, const SubGraphIds &);

} // namespace unwind
} // namespace memory
} // namespace poprithms

#endif
