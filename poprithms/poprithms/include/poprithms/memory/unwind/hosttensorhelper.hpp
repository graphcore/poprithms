// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_UNWIND_HOSTTENSORHELPER_HPP
#define POPRITHMS_MEMORY_UNWIND_HOSTTENSORHELPER_HPP

#include <map>

#include <poprithms/memory/chain/chain.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/path.hpp>
#include <poprithms/memory/unwind/valuedtensorid.hpp>
#include <poprithms/util/typedinteger.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

class Solution;

using common::multiout::TensorIds;
using nest::DisjointRegions;

class HostTensorHelper {

public:
  /** Unwind from Tensors in \a barriers to \a toSet, based on the
   * paths stored in the most recent call to setPaths. */
  static compute::host::Tensor
  get(const Solution &soln,
      const TensorId &toSet,
      const std::map<TensorId, compute::host::Tensor> &barriers);

  /**
   * A utility method for testing. For every Source and Barrier Tensor, create
   * a host::Tensor of distinct values. As an example, if a Graph has just 2
   * Sources and no Barriers, of Shapes (2,3) and (4) respectively, then the
   * return Tensors will be
   *  [[ 0 1 2 ]      and    [ 6 7 8 9 ].
   *   [ 3 4 5 ]]
   * */
  static std::map<TensorId, compute::host::Tensor>
  arangeBarriers(const Graph &);
};

} // namespace unwind
} // namespace memory
} // namespace poprithms

#endif
