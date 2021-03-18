// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

int main() {

  using namespace poprithms::memory::inplace;

  // Supoose the ML graph looks like:
  //
  //              input
  //              /.   \.
  //             slice slice (out-of-place slices)
  //             |     |
  //             sqrt  relu (out-of-place unary ops)
  //              \.  /.
  //           greaterThan
  //
  // The poprithms::memory::inplace::Graph will look like:
  //
  //          shape(10,)
  //           /.     \.
  //     slice[0:7]  slice[3:10]
  //         |         |
  //        aliasGate       aliasGate
  //         |         |
  //       unary      unary
  //         |         |
  //        aliasGate       aliasGate
  //           \.     /.
  //            noAlias
  //

  Graph g;

  // Add a variable Tensor to the graph.
  const auto var = Tensor::variable(g, {10});

  // Create slices followed by copies (closed aliasGatees)
  const auto slice0 = var.slice({0}, {7}).closedAliasGate();
  const auto slice1 = var.slice({3}, {10}).closedAliasGate();

  // Create the unary operations which act on the copied slices.
  const auto unary0 = slice0.modify().closedAliasGate();
  const auto unary1 = slice1.modify().closedAliasGate();

  // Create the operation (greaterThan) which we know will never create
  // aliases.
  Tensor::multi(g, {unary0, unary1}, {{7}}, {});

  std::cout << g << std::endl;

  // Tensors whose Op creators we want to try and inplace, in order of
  // attempt:
  Tensors toInplace{slice0, unary0, unary1, slice1};

  // Should we make sure not to make an Op inplace if it results in a Tensor
  // which is 1) constant, or 2) contains self-aliases, from being modified?

  const auto results =
      g.tryOpenings0(Tensor::opIds(toInplace), CheckParallelWriteable::Yes);

  std::cout << results << std::endl;

  std::cout << g << std::endl;

  // Now from g and results, we can apply the changes to the popart graph,
  // which should be simple.

  if (g.aliasGateIsOpen(slice1.opId())) {
    throw error("Expected the final inplacing attempt to fail");
  }

  // As a bonus, you can query aliasing information:
  std::cout << g.allAliases(var.id()) << std::endl;

  return 0;
}
