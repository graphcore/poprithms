// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_VANILLA_TYPES_HPP
#define POPRITHMS_SCHEDULE_VANILLA_TYPES_HPP

#include <vector>

namespace poprithms {
namespace schedule {
namespace vanilla {

enum class ErrorIfCycle {
  No = 0, ///< Do not error if there is a cycle, but return the partial
          ///< schedule of whatever could be scheduled.
  Yes     ///< Throw an error if there is a cycle.
};

enum class VerifyEdges {
  No = 0, ///< Do no checks that edges are valid
  Yes     ///< Check that edges are valid. In particular, check that all edges
          ///< terminate at valid nodes.
};

template <typename TNode> using Edges = std::vector<std::vector<TNode>>;

} // namespace vanilla
} // namespace schedule
} // namespace poprithms

#endif
