// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/memory/unwind/error.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/hosttensorhelper.hpp>
#include <poprithms/memory/unwind/solution.hpp>

int main() {

  using namespace poprithms::memory::unwind;
  using namespace poprithms::compute;

  Graph g;
  const auto sink = g.sink0({6});
  const auto s0   = g.slice(sink, 0, 3);
  const auto s1   = g.slice(sink, 3, 6);
  g.sumLike({s0, s1}, InIndex(0), 3.5);

  const auto s2     = g.slice(sink, 1, 4);
  const auto source = g.source0({3});
  g.insertValuedPair(source, s2, 65.);

  //
  //  ......
  //
  //  000...  slice 0
  //
  //  ...111  slice 1
  //
  //  .222..  slice 2
  //
  //  slice 2 should be enough to determine the
  //  layouts of slice 0 and slice 1.
  //

  const Solution soln(std::move(g));

  std::map<TensorId, host::Tensor> hosts;
  hosts.insert({source, host::Tensor::int64({3}, {7, 11, 13})});

  HostTensorHelper::get(soln, sink, hosts)
      .assertAllEquivalent(host::Tensor::int64({6}, {13, 7, 11, 13, 7, 11}));

  return 0;
}
