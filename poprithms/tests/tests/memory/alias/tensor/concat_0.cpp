// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poprithms/memory/alias/error.hpp>
#include <poprithms/memory/alias/graph.hpp>

namespace {
using namespace poprithms::memory::alias;
using namespace poprithms::memory::nest;

void testConcatFirst(const Shape &in0,
                     const Shape &in1,
                     const Shape &expected) {
  Graph g;

  const auto a   = g.tensor(g.allocate(in0));
  const auto b   = g.tensor(g.allocate(in1));
  const auto res = a.concatFirstDim({b}, 0);
  if (res.shape() != expected) {
    throw error("Unexpected shape on concatenated tensors. Should be " +
                expected.str() + ".");
  }
}

void testConcatFinal(const Shape &in0,
                     const Shape &in1,
                     const Shape &expected) {
  Graph g;

  const auto a   = g.tensor(g.allocate(in0));
  const auto b   = g.tensor(g.allocate(in1));
  const auto res = a.concatFinalDim({b}, 0);
  if (res.shape() != expected) {
    throw error("Unexpected shape on concatenated tensors. Should be " +
                expected.str() + ".");
  }
}

void test0() {
  // id  type         ins    shape  outs  aliases  aliased to
  // --- ------------ ------ ------ ----- -------- -----------
  // 0   Allocate(0)  ()     (3)    (2)   no       (0,2)
  // 1   Allocate(0)  ()     (4)    (2)   no       (1,2)
  // 2   Concat       (0,1)  (7)    ()    no       (0,1,2)
  testConcatFirst({3}, {4}, {7});
  // id  type         ins    shape  outs  aliases  aliased to
  // --- ------------ ------ ------ ----- -------- -----------
  // 0   Allocate(0)  ()     (3,1)  (2)   no       (0,2)
  // 1   Allocate(0)  ()     (4,1)  (2)   no       (1,2)
  // 2   Concat       (0,1)  (7,1)  ()    no       (0,1,2)
  testConcatFirst({3, 1}, {4, 1}, {7, 1});
  // id  type         ins    shape    outs  aliases  aliased to
  // --- ------------ ------ -------- ----- -------- -----------
  // 0   Allocate(0)  ()     (3,1,2)  (2)   no       (0,2)
  // 1   Allocate(0)  ()     (4,1,2)  (2)   no       (1,2)
  // 2   Concat       (0,1)  (7,1,2)  ()    no       (0,1,2)
  testConcatFirst({3, 1, 2}, {4, 1, 2}, {7, 1, 2});
}

void test1() {
  // id  type         ins    shape  outs  aliases  aliased to
  // --- ------------ ------ ------ ----- -------- -----------
  // 0   Allocate(0)  ()     (3)    (2)   no       (0,2)
  // 1   Allocate(0)  ()     (4)    (2)   no       (1,2)
  // 2   Concat       (0,1)  (7)    ()    no       (0,1,2)
  testConcatFinal({3}, {4}, {7});
  // id  type         ins    shape  outs  aliases  aliased to
  // --- ------------ ------ ------ ----- -------- -----------
  // 0   Allocate(0)  ()     (3,4)  (2)   no       (0,2)
  // 1   Allocate(0)  ()     (3,5)  (2)   no       (1,2)
  // 2   Concat       (0,1)  (3,9)  ()    no       (0,1,2)
  testConcatFinal({3, 4}, {3, 5}, {3, 9});
  // id  type         ins    shape    outs  aliases  aliased to
  // --- ------------ ------ -------- ----- -------- -----------
  // 0   Allocate(0)  ()     (3,1,2)  (2)   no       (0,2)
  // 1   Allocate(0)  ()     (3,1,4)  (2)   no       (1,2)
  // 2   Concat       (0,1)  (3,1,6)  ()    no       (0,1,2)
  testConcatFinal({3, 1, 2}, {3, 1, 4}, {3, 1, 6});
}
} // namespace

int main() {
  test0();
  test1();
  return 0;
}
