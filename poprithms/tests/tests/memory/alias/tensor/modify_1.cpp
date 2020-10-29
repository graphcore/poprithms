// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <map>
#include <set>

#include <poprithms/memory/alias/error.hpp>
#include <poprithms/memory/alias/graph.hpp>
namespace {
using namespace poprithms::memory::alias;
void testToIdentity0() {

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
  //         in1   dst - out1   .
  //                    \       .
  //          in2 - foo  out2   .
  //

  Graph g;
  const auto in0_  = g.tensor(g.allocate({4, 8}));
  const auto src_  = in0_.reverse(1);
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

  std::cout << g << std::endl;
  g.confirmAllAliasesMap(expectedAliases0);

  dst_.toIdentityFrom(src_);

  //   in0 - src - out0          .
  //          \                  .
  //   in1    dst - out1         .
  //             \               .
  //   in2 - foo  out2           .
  //

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

void testConcat0() {

  // From :
  //
  // x0-|
  //    |--y0--|
  // x1-|      |
  //           |--z
  // x2-|      |
  //    |--y1--|
  // x3-|
  //
  // where the merges are concatenations, to
  //
  // x0
  //       y0--|
  // x1        |
  //           |--z
  // x2        |
  //       y1--|
  // x3.
  //
  // That is, convert the first concatenations to allocations.
  //

  Graph g;
  std::vector<TensorId> xs;
  std::vector<TensorId> ys;
  for (uint64_t i = 0; i < 4; ++i) {
    // Allocations of Shape (3,5)
    xs.push_back(g.allocate({3, 5}));
    if (i % 2 == 1) {
      ys.push_back(g.concat({xs[i - 1], xs[i]}, 0));
    }
  }
  /*z = */
  const auto z = g.concat(ys, 0);

  auto gPreModifications = g;

  // Convert ys to allocations:
  for (auto y : ys) {
    g.toAllocation(y, 0);
  }

  const auto gWithYsAsAllocs0 = g;

  if (gWithYsAsAllocs0 == gPreModifications) {
    throw error("conversion to allocations had no effect - incorrect");
  }

  std::map<TensorId, std::set<TensorId>> expectedAliases{
      {xs[0], {xs[0]}},
      {xs[1], {xs[1]}},
      {xs[2], {xs[2]}},
      {xs[3], {xs[3]}},
      {ys[0], {ys[0], z}},
      {ys[1], {ys[1], z}},
      {z, {ys[0], ys[1], z}}};

  gWithYsAsAllocs0.confirmAllAliasesMap(expectedAliases);

  // Convert back to concats:
  uint64_t i = 1;
  for (auto y : ys) {
    g.allocationToConcat({xs[i - 1], xs[i]}, 0, y);
    i += 2;
  }

  const auto gRevertedToOrigins = g;

  if (gPreModifications != gRevertedToOrigins) {
    std::ostringstream oss;
    oss << "Converting to allocations, then back to concats, "
        << "should result in the same Graph as the initial one.";
    throw error(oss.str());
  }

  // Convert ys to allocations again:
  for (auto y : ys) {
    g.toAllocation(y, 0);
  }

  const auto gWithYsAsAllocs1 = g;

  for (auto x : xs) {
    if (!g.ins(x).empty() || !g.outs(x).empty()) {
      throw error("x allocations have no consumers");
    }
  }
  for (auto y : ys) {
    if (!g.ins(y).empty() || g.outs(y) != std::vector{z}) {
      throw error("y allocations have z consumer");
    }
  }
  if (g.ins(z).size() != 2 || !g.outs(z).empty()) {
    throw error("z has 2 inputs and 0 outputs");
  }

  if (gWithYsAsAllocs0 != gWithYsAsAllocs1) {
    std::ostringstream oss;
    oss << "Converting to allocations, then back to concats, "
        << "then to allocations, "
        << "should result in the same Graph as the initial "
        << "conversion to allocations.";
    throw error(oss.str());
  }
}

//  id  type      ins      shape  outs  aliases  aliased to
//  --- --------- -------- ------ ----- -------- -----------
//  0   Allocate  ()       (2,3)  (1)   no       (0,1)
//  1   Concat    (0,0,0)  (6,3)  ()    yes      (0,1)
void testConcat1() {
  Graph g;
  const auto x0 = g.allocate({2, 3}, 0);
  const auto x1 = g.allocate({6, 3}, 0);
  g.allocationToConcat({x0, x0, x0}, 0, x1);

  if (!g.containsAliases(x1)) {
    throw error("x1 does contain aliases");
  }

  Graph g2;
  const auto x2 = g2.allocate({2, 3}, 0);
  g2.concat({x2, x2, x2}, 0);

  if (g != g2) {
    throw error("Error in testConcat1");
  }
}

} // namespace

int main() {
  testToIdentity0();
  testConcat0();
  testConcat1();
  return 0;
}
