// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>

int main() {

  using namespace poprithms::memory::inplace;

  // Supoose the ML graph looks like:
  //
  //              input
  //              /.   \.
  //             slice slice
  //             |     |
  //             sqrt  relu
  //              \.  /.
  //             greaterThan
  //
  // The poprithms::memory::inplace::Graph will look like:
  //
  //          shape(10,)
  //           /.     \.
  //     slice[0:7]  slice[3:10]
  //          |        |
  //       unary      unary
  //           \.     /.
  //            noAlias
  //
  // The poprithms::memory::inplace::Graph is made up of only a Ops:
  //
  // The usual view-changing Ops:
  //   dimShuffle, reverse, slice, concat, reshape, etc.
  //
  // Modifiers:
  //   unary, binary
  //
  // General purpose never-inplacing Op, for things like matmul, batch-norm:
  //   noAlias.
  //
  // Let's construct the Graph. Currently there is no Tensor class, so we use
  // the "builder" approach (see T29193 for Tensor class)

  Graph g;

  // Add a variable Tensor to the graph.
  const auto varId = g.variable({10});

  // To add a constant, use g.constant({10}).

  // Create the out-of-place slices.
  const auto slice0 = g.slice(varId, AliasType::outplace(), {0}, {7});
  const auto slice1 = g.slice(varId, AliasType::outplace(), {3}, {10});

  // Create the unary operations which act on the slices.
  const auto unary0 = g.unary(slice0, AliasType::outplace());
  const auto unary1 = g.unary(slice1, AliasType::outplace());

  // Create the operation (greaterThan) which we know will never create
  // aliases.
  g.noAlias({unary0, unary1}, {{7}});

  std::cout << g << std::endl;
  // clang-format off
  //
  // OpId  OpType                                             InTensors  InOps  OutIndex  TensorId  Shape  TensorType   Aliases  Constants  AliasedTo  
  // ----- -------------------------------------------------- ---------- ------ --------- --------- ------ ------------ -------- ---------- ---------- 
  // 0     Alloc(color=1)                                     ()         ()     0         0         (10)   Allocate(1)  no       no         (0)        
  // 1     SettSample(region=(shape=(10),setts=(((7,3,0)))))  (0)        (0)    0         1         (7)    Allocate(1)  no       no         (1)        
  // 2     SettSample(region=(shape=(10),setts=(((7,3,3)))))  (0)        (0)    0         2         (7)    Allocate(1)  no       no         (2)        
  // 3     Unary                                              (1)        (1)    0         3         (7)    Allocate(1)  no       no         (3)        
  // 4     Unary                                              (2)        (2)    0         4         (7)    Allocate(1)  no       no         (4)        
  // 5     NoAlias                                            (3,4)      (3,4)  0         5         (7)    Allocate(1)  no       no         (5)
  //
  // clang-format on

  //               input
  //              /.    \.
  //             slice0  slice1
  //             |       |
  //             unary0  unary1
  //              \.    /.
  //             greaterThan

  // Tensors whose Op creators we want to try and inplace, in order of
  // attempt:
  TensorIds toInplace{slice0, slice1, unary0, unary1};

  // Should we make sure not to make an Op inplace if it results in a Tensor
  // which is 1) constant, or 2) contains self-aliases, from being modified?

  const auto results =
      g.tryInplaces(Graph::createProposalsAllInplace(toInplace),
                    CheckParallelWriteable::Yes);

  std::cout << results << std::endl;
  //  Applied(()),
  //  Applied(()),
  //  Applied(((1->3),(2->3),(4->3))),
  //  Cycle.

  std::cout << g << std::endl;
  // clang-format off
  //
  //  OpId  OpType                                             InTensors  InOps    OutIndex  TensorId  Shape  TensorType              Aliases  Constants  AliasedTo  
  //  ----- -------------------------------------------------- ---------- -------- --------- --------- ------ ----------------------- -------- ---------- ---------- 
  //  0     Alloc(color=1)                                     ()         ()       0         0         (10)   Allocate(1)             no       no         (0,1,2,3)  
  //  1     SettSample(region=(shape=(10),setts=(((7,3,0)))))  (0)        (0)      0         1         (7)    SettSample (((7,3,0)))  no       no         (0,1,2,3)  
  //  2     SettSample(region=(shape=(10),setts=(((7,3,3)))))  (0)        (0)      0         2         (7)    SettSample (((7,3,3)))  no       no         (0,1,2,3)  
  //  3     Unary                                              (1)        (1,2,4)  0         3         (7)    Identity                no       no         (0,1,2,3)  
  //  4     Unary                                              (2)        (2)      0         4         (7)    Allocate(1)             no       no         (4)        
  //  5     NoAlias                                            (3,4)      (3,4)    0         5         (7)    Allocate(1)             no       no         (5)        
  //
  // clang-format on
  //

  // Now from g and results, we can apply the changes to the popart graph,
  // which should be simple.

  if (g.aliasType(unary1) != AliasType::outplace()) {
    throw error("Expected the final inplacing attempt to fail");
  }

  // As a bonus, you can query aliasing information:
  std::cout << g.allAliases(varId) << std::endl;
  // "Op, OutIndex":
  // ((0,0),(1,0),(2,0),(3,0))
  // These are the outputs of varId, slice0, slice1, and unary0.
  return 0;
}
