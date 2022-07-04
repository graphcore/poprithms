// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_PRUNE_PRUNER_HPP
#define POPRITHMS_COMMON_COMPUTE_PRUNE_PRUNER_HPP

#include <map>
#include <memory>
#include <set>
#include <vector>

#include <poprithms/common/compute/graph.hpp>

namespace poprithms {
namespace common {
namespace compute {

/**
 * Prune a compute::Graph.
 *
 * Note that control deps are not transferred while pruning.
 * */
class Pruner {
public:
  /**
   * Do not prune any tensors in #retain. All other tensors can
   * be pruned if they are determined to not have any effect on the tensors in
   * #retain.
   * */
  static void prune(Graph &g, const TensorIds &retain) {
    pruneButPreserveUnpruneableRefs(g, retain);
  }

  /**
   * Do not prune any host tensors.
   * */
  static void preserveHostTensors(Graph &g) { prune(g, g.hostTensorIds()); }

private:
  static TensorIds getUnprunenableRefs(const Graph &);
  static void pruneButPreserveUnpruneableRefs(Graph &, TensorIds retain);
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
