// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <map>
#include <set>

#include <poprithms/memory/alias/error.hpp>
#include <poprithms/memory/alias/graph.hpp>

void test0() {

  using namespace poprithms::memory::alias;

  //
  //           bar   out0         .
  //          /    /              .
  //      in0 - id - out1         .
  //      in1 /                   .
  //          \                   .
  //           foo                .
  //                              .
  // to                           .
  //                              .
  //          bar       out0      .
  //         /        /           .
  //     in0       id - out1      .
  //     in1                      .
  //         \                    .
  //          foo                 .

  Graph g;
  auto in0_       = g.tensor(g.allocate({3, 5}));
  const auto in1_ = g.tensor(g.allocate({4, 5}));

  const auto bar_ = in0_.reshape({5, 3});
  const auto foo_ = in1_.reshape({20, 1});
  auto id_        = concat({in0_, in1_}, 0);

  const auto out0_ = id_.dimShuffle({{1, 0}});
  const auto out1_ = id_.flatten();

  const auto in0  = in0_.id();
  const auto in1  = in1_.id();
  const auto bar  = bar_.id();
  const auto foo  = foo_.id();
  const auto id   = id_.id();
  const auto out0 = out0_.id();
  const auto out1 = out1_.id();

  //                              .
  //           bar   out0         .
  //          /    /              .
  //      in0 - id - out1         .
  //      in1 /                   .
  //          \                   .
  //           foo                .
  //
  //  id  type           ins    shape   outs   aliases  aliased to
  //  --- -------------- ------ ------- ------ -------- ----------------
  //  0   Allocate       ()     (3,5)   (2,4)  no       (0,2,4,5,6)
  //  1   Allocate       ()     (4,5)   (3,4)  no       (1,3,4,5,6)
  //  2   Reshape        (0)    (5,3)   ()     no       (0,2,4,5,6)
  //  3   Reshape        (1)    (20,1)  ()     no       (1,3,4,5,6)
  //  4   Concat         (0,1)  (7,5)   (5,6)  no       (0,1,2,3,4,5,6)
  //  5   Permute (1,0)  (4)    (5,7)   ()     no       (0,1,2,3,4,5,6)
  //  6   Reshape        (4)    (35)    ()     no       (0,1,2,3,4,5,6)
  std::map<TensorId, std::set<TensorId>> expectedAliases0{
      {in0, {in0, bar, id, out0, out1}},
      {bar, {in0, bar, id, out0, out1}},
      {in1, {in1, foo, id, out0, out1}},
      {foo, {in1, foo, id, out0, out1}},
      {id, {in0, in1, foo, bar, id, out0, out1}},
      {out0, {in0, in1, foo, bar, id, out0, out1}},
      {out1, {in0, in1, foo, bar, id, out0, out1}},
  };

  g.confirmAllAliasesMap(expectedAliases0);

  id_.toAllocation(Color(7));

  //                             .
  //          bar       out0     .
  //         /        /          .
  //     in0       id - out1     .
  //     in1                     .
  //         \                   .
  //          foo                .
  //
  //  id  type           ins  shape   outs   aliases  aliased to
  //  --- -------------- ---- ------- ------ -------- -----------
  //  0   Allocate       ()   (3,5)   (2)    no       (0,2)
  //  1   Allocate       ()   (4,5)   (3)    no       (1,3)
  //  2   Reshape        (0)  (5,3)   ()     no       (0,2)
  //  3   Reshape        (1)  (20,1)  ()     no       (1,3)
  //  4   Allocate       ()   (7,5)   (5,6)  no       (4,5,6)
  //  5   Permute (1,0)  (4)  (5,7)   ()     no       (4,5,6)
  //  6   Reshape        (4)  (35)    ()     no       (4,5,6)
  std::map<TensorId, std::set<TensorId>> expectedAliases1{
      {in0, {in0, bar}},
      {bar, {bar, in0}},
      {in1, {in1, foo}},
      {foo, {foo, in1}},
      {id, {id, out0, out1}},
      {out0, {out0, id, out1}},
      {out1, {out1, id, out0}}};

  g.confirmAllAliasesMap(expectedAliases1);

  in0_.toAllocation(Color(11));

  g.confirmAllAliasesMap(expectedAliases1);

  if (!in0_.containsColor(Color(11)) || in0_.containsColor(Color(0))) {
    throw error("Color not updated correctly");
  }
}

int main() {
  test0();
  return 0;
}
