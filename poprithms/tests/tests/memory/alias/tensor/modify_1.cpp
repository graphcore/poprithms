// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <map>
#include <set>

#include <poprithms/memory/alias/error.hpp>
#include <poprithms/memory/alias/graph.hpp>

void test0() {

  using namespace poprithms::memory::alias;

  // from                       .
  //                            .
  //       in0 - src - out0     .
  //                            .
  //        in1 - dst - out1    .
  //             /     \        .
  //         in2 - foo  out2    .
  //                            .
  // to                         .
  //                            .
  //        in0 - src - out0    .
  //               \            .
  //        in1    dst - out1   .
  //                  \         .
  //        in2 - foo  out2     .
  //

  Graph g;
  const auto in0_  = g.tensor(g.allocate({4, 8}));
  const auto src_  = in0_.dimshuffle({{1, 0}});
  const auto out0_ = src_.flatten();
  const auto in1_  = g.tensor(g.allocate({4, 5}));
  const auto in2_  = g.tensor(g.allocate({4, 3}));
  auto dst_        = concat({in1_, in2_}, 1);
  const auto out1_ = dst_.flatten();
  const auto foo_  = in2_.slice({0, 0}, {4, 2});
  const auto out2_ = dst_.reverse(1);

  const auto in0  = in0_.id();
  const auto src  = src_.id();
  const auto out0 = out0_.id();
  const auto in1  = in1_.id();
  const auto in2  = in2_.id();
  const auto dst  = dst_.id();
  const auto out1 = out1_.id();
  const auto foo  = foo_.id();
  const auto out2 = out2_.id();

  std::map<TensorId, std::set<TensorId>> expectedAliases0{
      {in0, {in0, src, out0}},
      {src, {in0, src, out0}},
      {out0, {in0, src, out0}},
      {in1, {in1, dst, out1, out2}},
      {in2, {in2, dst, out1, foo, out2}},
      {dst, {in1, in2, dst, out1, foo, out2}},
      {out1, {in1, in2, dst, out1, foo, out2}},
      {foo, {in2, dst, out1, foo, out2}},
      {out2, {in1, in2, dst, out1, foo, out2}}};

  //   in0 - src - out0       .
  //                          .
  //    in1 - dst - out1      .
  //         /     \          .
  //     in2 - foo  out2      .
  //
  //  id type                      ins     shape   outs   aliases
  //
  //  0  Allocate                  ()     (3,4)   (1)    no    (0,1,2)
  //  1  Permute (1,0)             (0)    (4,3)   (2)    no    (0,1,2)
  //  2  Reshape                   (1)    (12)    ()     no    (0,1,2)
  //  3  Allocate                  ()     (4,5)   (5)    no    (3,5,6,8)
  //  4  Allocate                  ()     (4,11)  (5,7)  no    (4,5,6,7,8)
  //  5  Concat                    (3,4)  (4,16)  (6,8)  no    (3,4,5,6,7,8)
  //  6  Reshape                   (5)    (32,2)  ()     no    (3,4,5,6,7,8)
  //  7  SettSample (()((2,9,0)))  (4)    (4,2)   ()     no    (4,5,6,7,8)
  //  8  Reverse (1)               (5)    (4,16)  ()     no    (3,4,5,6,7,8)
  //

  std::cout << g << std::endl;
  g.confirmAllAliasesMap(expectedAliases0);

  dst_.toIdentityFrom(src_);

  //   in0 - src - out0          .
  //          \                  .
  //   in1    dst - out1         .
  //             \               .
  //   in2 - foo  out2           .
  //
  // id  type                      ins  shape  outs   aliases  aliased to
  // --- ------------------------- ---- ------ ------ -------- --------------
  // 0   Allocate                  ()   (4,8)  (1)    no       (0,1,2,5,6,8)
  // 1   Permute (1,0)             (0)  (8,4)  (2,5)  no       (0,1,2,5,6,8)
  // 2   Reshape                   (1)  (32)   ()     no       (0,1,2,5,6,8)
  // 3   Allocate                  ()   (4,5)  ()     no       (3)
  // 4   Allocate                  ()   (4,3)  (7)    no       (4,7)
  // 5   Reverse ()                (1)  (4,8)  (6,8)  no       (0,1,2,5,6,8)
  // 6   Reshape                   (5)  (32)   ()     no       (0,1,2,5,6,8)
  // 7   SettSample (()((2,1,0)))  (4)  (4,2)  ()     no       (4,7)
  // 8   Reverse (1)               (5)  (4,8)  ()     no       (0,1,2,5,6,8)

  std::cout << g << std::endl;
  std::map<TensorId, std::set<TensorId>> expectedAliases1{
      {in0, {in0, src, out0, dst, out1, out2}},
      {src, {in0, src, out0, dst, out1, out2}},
      {out0, {in0, src, out0, dst, out1, out2}},
      {out1, {in0, src, out0, dst, out1, out2}},
      {out2, {in0, src, out0, dst, out1, out2}},
      {dst, {in0, src, out0, dst, out1, out2}},
      {in1, {in1}},
      {in2, {in2, foo}},
      {foo, {in2, foo}}};

  g.confirmAllAliasesMap(expectedAliases1);
}

int main() {
  test0();
  return 0;
}
