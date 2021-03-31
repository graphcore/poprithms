// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/alias/error.hpp>
#include <poprithms/memory/alias/graph.hpp>

int main() {

  using namespace poprithms::memory::alias;
  using namespace poprithms::memory::nest;

  Graph g;

  // 000
  // 000
  //  .
  //  .
  // 000
  // 000
  const auto alloc0 = g.tensor(g.allocate({10, 3}));

  // 11
  // 11
  // .
  // .
  // 11
  // 11
  const auto alloc1 = g.tensor(g.allocate({10, 2}));

  // 2222222222
  // 2222222222
  // 2222222222
  const auto alloc2 = g.tensor(g.allocate({3, 10}));

  // 00011
  // 00011
  // 00011
  // 00011
  // 00011
  // 00011
  // 00011
  // 00011
  // 00011
  // 00011
  const auto cat = concat({alloc0, alloc1}, 1);

  // 0001100011
  // 0001100011
  // 0001100011
  // 0001100011
  // 0001100011
  const auto rsh = cat.reshape({5, 10});

  // 2222222222
  // 2222222222
  // 2222222222
  // 0001100011
  // 0001100011
  // 0001100011
  // 0001100011
  // 0001100011
  const auto cat2 = concat({alloc2, rsh}, 0);

  // 222222
  // 222222
  // 011000
  // 011000
  // 011000
  // 011000
  const auto slc = cat2.slice({1, 2}, {7, 8});

  // reverse in both axes:
  //
  // 000110
  // 000110
  // 000110
  // 000110
  // 222222
  // 222222
  const auto flp = slc.reverse({0, 1});

  // shuffle the dimensions:
  //
  // 000022
  // 000022
  // 000022
  // 111122
  // 111122
  // 000022
  const auto prm = flp.dimShuffle({{1, 0}});

  // 000022000022000022111122111122000022
  const auto flat = prm.flatten();
  if (flat.numElements() != 36) {
    throw error("Expected 36 elements in final flattened Tensor");
  }

  // logging string:
  std::cout << g.verboseString() << std::endl;

  //
  // clang-format off
  // 
  //   id  type                             ins    shape   outs  aliases  aliased to
  //   --- -------------------------------- ------ ------- ----- -------- ----------------------
  //   0   Allocate                         ()     (10,3)  (3)   no       (0,3,4,5,6,7,8,9)
  //   1   Allocate                         ()     (10,2)  (3)   no       (1,3,4,5,6,7,8,9)
  //   2   Allocate                         ()     (3,10)  (5)   no       (2,5,6,7,8,9)
  //   3   Concat                           (0,1)  (10,5)  (4)   no       (0,1,3,4,5,6,7,8,9)
  //   4   Reshape                          (3)    (5,10)  (5)   no       (0,1,3,4,5,6,7,8,9)
  //   5   Concat                           (2,4)  (8,10)  (6)   no       (0,1,2,3,4,5,6,7,8,9)
  //   6   SettSample (((6,2,1))((6,4,2)))  (5)    (6,6)   (7)   no       (0,1,2,3,4,5,6,7,8,9)
  //   7   Reverse (0,1)                    (6)    (6,6)   (8)   no       (0,1,2,3,4,5,6,7,8,9)
  //   8   Permute (1,0)                    (7)    (6,6)   (9)   no       (0,1,2,3,4,5,6,7,8,9)
  //   9   Reshape                          (8)    (36)    ()    no       (0,1,2,3,4,5,6,7,8,9)
  //
  //   Origins:
  //  
  //   id  regions aliased in allocation Tensors
  //   --  -------------------------------------
  //   0:  [0]:(shape=(10,3),setts=(((),()))))
  //
  //   1:  [1]:(shape=(10,2),setts=(((),()))))
  //
  //   2:  [2]:(shape=(3,10),setts=(((),()))))
  //
  //   3:  [0]:(shape=(10,3),setts=(((),()))))
  //       [1]:(shape=(10,2),setts=(((),()))))
  //
  //   4:  [0]:(shape=(10,3),setts=(((),()))))
  //       [1]:(shape=(10,2),setts=(((),()))))
  //
  //   5:  [0]:(shape=(10,3),setts=(((),()))))
  //       [1]:(shape=(10,2),setts=(((),()))))
  //       [2]:(shape=(3,10),setts=(((),()))))
  //
  //   6:  [0]:(shape=(10,3),setts=((((8,2,0)(1,1,0)),((1,2,2)))(((8,2,0)(1,1,1)),()))))
  //       [1]:(shape=(10,2),setts=((((8,2,0)(1,1,0)),()))))
  //       [2]:(shape=(3,10),setts=((((2,1,1)),((6,4,2))))))
  //
  //   7:  [0]:(shape=(10,3),setts=((((8,2,0)(1,1,0)),((1,2,2)))(((8,2,0)(1,1,1)),()))))
  //       [1]:(shape=(10,2),setts=((((8,2,0)(1,1,0)),()))))
  //       [2]:(shape=(3,10),setts=((((2,1,1)),((6,4,2))))))
  //
  //   8:  [0]:(shape=(10,3),setts=((((8,2,0)(1,1,0)),((1,2,2)))(((8,2,0)(1,1,1)),()))))
  //       [1]:(shape=(10,2),setts=((((8,2,0)(1,1,0)),()))))
  //       [2]:(shape=(3,10),setts=((((2,1,1)),((6,4,2))))))
  //
  //   9:  [0]:(shape=(10,3),setts=((((8,2,0)(1,1,0)),((1,2,2)))(((8,2,0)(1,1,1)),()))))
  //       [1]:(shape=(10,2),setts=((((8,2,0)(1,1,0)),()))))
  //       [2]:(shape=(3,10),setts=((((2,1,1)),((6,4,2))))))
  //
  // clang-format on

  return 0;
}
