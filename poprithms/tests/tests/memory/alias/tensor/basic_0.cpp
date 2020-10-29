// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>

#include <poprithms/memory/alias/error.hpp>
#include <poprithms/memory/alias/graph.hpp>

int main() {

  using namespace poprithms::memory::alias;
  using namespace poprithms::memory::nest;

  Graph g;

  //                alloc \.
  //                alloc - concat - broadcast
  //                alloc /.              \.
  //  alloc                              subsample ---- flatten
  //    \                   .                                  \.
  //   slice                             settsample ---------  cat
  //     \  \                                |                  |
  //     vstack                            expand               |
  //      \   \                              |                 out
  //      hstack                          reverse
  //       \   \                             |
  //       concat -- flatten  - reshape -- dimshuffle
  //

  // clang-format off
  //
  // id  type                               ins            shape      outs     aliases  
  // --- ---------------------------------- -------------- ---------- -------- -------- 
  // 0   Allocate                           ()             (2,3,4)    (1)      no       
  // 1   SettSample (()((2,1,1))((2,2,1)))  (0)            (2,2,2)    (2)      no       
  // 2   Concat                             (1,1)          (2,2,4)    (3)      yes      
  // 3   Concat                             (2,2)          (4,2,4)    (4)      yes      
  // 4   Concat                             (3,3)          (4,4,4)    (5)      yes      
  // 5   Reshape                            (4)            (64)       (6)      yes      
  // 6   Reshape                            (5)            (64,1)     (7)      yes      
  // 7   Permute (1,0)                      (6)            (1,64)     (8)      yes      
  // 8   Reverse (0)                        (7)            (1,64)     (9)      yes      
  // 9   Expand                             (8)            (10,64)    (10)     yes      
  // 10  SettSample (((1,1,0))((1,1,0)))    (9)            (5,32)     (14,20)  yes      
  // 11  Allocate                           ()             (5,32)     (14)     no       
  // 12  Allocate                           ()             (5,32)     (14)     no       
  // 13  Allocate                           ()             (5,32)     (14)     no       
  // 14  Concat                             (11,12,10,13)  (5,128)    (15)     yes      
  // 15  Reshape                            (14)           (1,5,128)  (16)     yes      
  // 16  Expand                             (15)           (3,5,128)  (17)     yes      
  // 17  Reshape                            (16)           (15,128)   (18)     yes      
  // 18  SettSample (((1,2,0))())           (17)           (5,128)    (19)     yes      
  // 19  Reshape                            (18)           (640)      (21)     yes      
  // 20  Reshape                            (10)           (160)      (21)     yes      
  // 21  Concat                             (20,19)        (800)      ()       yes      
  //
  // clang-format on

  const auto alloc0 = g.allocate({2, 3, 4});

  const auto arr0  = g.tensor(alloc0);
  const auto arr1  = arr0.slice({0, 1, 1}, {2, 3, 3});
  const auto arr2  = arr1.vstack({arr1}, 0);
  const auto arr3  = arr2.hstack({arr2}, 1);
  const auto arr5  = arr3.concat({arr3}, 0, 1).flatten();
  const auto shape = arr5.shape();
  const auto arr8  = arr5.reshape({shape.nelms(), 1})
                        .dimshuffle({{1, 0}})
                        .reverse(std::vector<uint64_t>(1, 0));

  auto sh = arr8.shape().get();
  std::replace(sh.begin(), sh.end(), 1, 10);

  const auto arr9  = arr8.expand(sh);
  const auto arr10 = arr9.settsample(
      {arr9.shape(), std::vector<Sett>(arr9.rank_u64(), {{{1, 1, 0}}})});

  const auto alloc1 = g.tensor(g.allocate(arr10.shape()));
  const auto alloc2 = g.tensor(g.allocate(arr10.shape()));
  const auto alloc3 = g.tensor(g.allocate(arr10.shape()));
  const auto arr15  = arr10.concat({alloc1, alloc2, alloc3}, 2, 1)
                         .broadcast(3, 0)
                         .subsample(3, 0)
                         .flatten();

  auto out = g.concat({arr10.flatten().id(), arr15.id()}, 0);

  if (g.tensor(out).numElements() != 800) {
    throw error("Expected 800 elements in final Tensor (see log)");
  }

  const auto r0 = Region::fromBounds({4, 3, 4}, {0, 0, 0}, {2, 3, 4});
  const auto r1 = Region::fromBounds({4, 3, 4}, {2, 0, 0}, {4, 3, 4});
  g.settfill({alloc0, alloc0}, DisjointRegions{Shape({4, 3, 4}), {r0, r1}});

  std::cout << g << std::endl;

  return 0;
}
