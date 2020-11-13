// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>

namespace {

using namespace poprithms::memory::inplace;
void testReverse0() {

  Graph g;
  const auto v0 = g.variable({10});
  const auto r0 = g.reverse(v0, AliasType::outplace(), {0});

  for (int64_t sliceSize : {3, 5, 7}) {
    auto g0       = g;
    const auto s0 = g0.slice(v0, AliasType::outplace(), {0}, {sliceSize});
    const auto s1 = g0.slice(r0, AliasType::outplace(), {0}, {sliceSize});

    g0.unary(s0, AliasType::allInplace());
    g0.unary(s1, AliasType::allInplace());
    g0.tryInplaces(Graph::createProposalsAllInplace({s0, s1, r0}),
                   CheckParallelWriteable::Yes);
    if (g0.aliasType(s0) != AliasType::allInplace() ||
        g0.aliasType(s1) != AliasType::allInplace()) {
      throw error("slices should both be inplace");
    }

    bool expectReverseInplace = sliceSize <= 5;

    if ((g0.aliasType(r0) == AliasType::allInplace()) !=
        expectReverseInplace) {
      throw error("expect reverse inplace iff sliceSize <= 5");
    }
  }
}

void testReshape0() {

  // Using that 14 and 5 are co-prime here, which guarantees vertical slices
  // (post reshape) always intersect with horizontal slices (pre reshape).

  Graph g;
  auto v0 = g.variable({14, 5});

  // x . .
  // x . .          x . . x
  // x . .    ==>   . . x .
  // x . .          . x . .
  //

  auto s0  = g.slice(v0, AliasType::outplace(), {0, 2}, {14, 3});
  auto nl0 = g.unary(s0, AliasType::outplace());

  auto r0 = g.reshape(v0, AliasType::outplace(), {5, 14});

  auto s1  = g.slice(r0, AliasType::outplace(), {0, 3}, {5, 4});
  auto nl1 = g.unary(s1, AliasType::outplace());

  auto s2  = g.slice(r0, AliasType::outplace(), {0, 11}, {5, 12});
  auto nl2 = g.unary(s2, AliasType::outplace());

  const auto &gBase = g;
  auto test         = [&gBase](const TensorIds &ids,
                       const std::vector<AliasType> &expected) {
    auto g2 = gBase;
    g2.tryInplaces(Graph::createProposalsAllInplace(ids),
                   CheckParallelWriteable::Yes);
    for (uint64_t i = 0; i < ids.size(); ++i) {
      if (g2.aliasType(ids[i]) != expected[i]) {
        std::ostringstream oss;
        oss << "With initial Graph " << gBase << ", final Graph is " << g2
            << ".";
        throw error(oss.str());
      }
    }
  };

  const auto I = AliasType::allInplace();
  const auto O = AliasType::outplace();

  //               v0
  //             /  |
  //          s0    r0 - s1 - nl1
  //        /       |
  //     nl0        s2 - nl2
  //

  test({nl0, s0, s1, nl1, s2, nl2, r0}, //
       {I, I, I, I, I, I, O});

  test({r0, nl0, s0, s1, nl1, s2, nl2}, //
       {I, I, I, I, O, I, O});

  test({nl0, s1, nl1, r0, s2, nl2, s0}, //
       {I, I, I, I, I, I, O});
}

void testEmptySlice0() {

  Graph g;
  auto a  = g.variable({10, 10});
  auto b  = g.slice(a, AliasType::outplace(), {0, 0}, {10, 0});
  auto c  = g.reverse(b, AliasType::outplace(), {1});
  auto d  = g.dimShuffle(c, AliasType::outplace(), {{0, 1}});
  auto e  = g.reshape(d, AliasType::outplace(), {1, 0});
  auto f  = g.unary(e, AliasType::outplace());
  auto g_ = g.unary(e, AliasType::outplace());
  g.tryInplaces(Graph::createProposalsAllInplace({b, c, d, e, f, g_}),
                CheckParallelWriteable::Yes);
  std::cout << g << std::endl;
  for (auto id : {b, c, d, e, f, g_}) {
    if (g.aliasType(id) != AliasType::allInplace()) {
      throw error("Failed to inplace all in testEmptySlice0");
    }
  }
}

} // namespace

int main() {

  testReverse0();
  testReshape0();
  testEmptySlice0();
  return 0;
}
