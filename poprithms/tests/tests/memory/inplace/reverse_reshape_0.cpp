// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/graph.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

namespace {

using namespace poprithms::memory::inplace;
void testReverse0() {
  //
  //       "2"                     "1"
  //  v0 - aliasGate - reverse - slice - aliasGate - unary
  //  |
  // slice - aliasGate - unary
  //         "0"
  //
  // Depending on slice width, "2" will be opened.
  //

  for (int64_t sliceSize : {3, 5, 7}) {

    Graph g;
    const auto v0              = Tensor::variable(g, {10});
    const auto v0AliasGate     = v0.slice({0}, {sliceSize}).closedAliasGate();
    const auto preRevAliasGate = v0.closedAliasGate();
    const auto postRevAliasGate =
        preRevAliasGate.reverse(0).slice({0}, {sliceSize}).closedAliasGate();
    v0AliasGate.modify();
    postRevAliasGate.modify();

    // "0", "1", "2":
    OpIds aliasGates{
        v0AliasGate.opId(), postRevAliasGate.opId(), preRevAliasGate.opId()};
    g.tryOpenings0(
        aliasGates, CheckParallelWriteable::Yes, AllowMultiGateAlias::No);

    if (v0AliasGate.aliasGateIsClosed() ||
        postRevAliasGate.aliasGateIsClosed()) {
      throw error("slices should both be inplace");
    }

    bool expectReverseInplace = sliceSize <= 5;

    if (preRevAliasGate.aliasGateIsOpen() != expectReverseInplace) {
      throw error("expect reverse inplace iff sliceSize <= 5");
    }
  }
}

void testReshape0() {

  // Using that 14 and 5 are co-prime here, which guarantees vertical slices
  // (post reshape) always intersect with horizontal slices (pre reshape).

  Graph g;
  auto v0 = Tensor::variable(g, {14, 5});

  // x . .
  // x . .          x . . x
  // x . .    ==>   . . x .
  // x . .          . x . .

  //               v0
  //             /  |
  //          s0    r0 - s1 - nl1
  //        /       |
  //     nl0        s2 - nl2
  //

  auto s0 = v0.slice({0, 2}, {14, 3}).closedAliasGate();
  s0.modify();

  auto r0 = v0.reshape({5, 14}).closedAliasGate();

  auto s1 = r0.slice({0, 3}, {5, 4}).closedAliasGate();
  s1.modify();

  auto s2 = r0.slice({0, 11}, {5, 12}).closedAliasGate();
  s2.modify();

  const auto &gBase = g;
  auto test         = [&gBase](const TensorIds &ts,
                       const std::vector<bool> &expectedOpen) {
    auto g2 = gBase;
    g2.tryOpenings0(ts, CheckParallelWriteable::Yes, AllowMultiGateAlias::No);
    for (uint64_t i = 0; i < ts.size(); ++i) {
      if (g2.aliasGateIsOpen(ts[i].opId()) != expectedOpen[i]) {
        std::ostringstream oss;
        oss << "With initial Graph " << gBase << ", final Graph is " << g2
            << ".";
        throw error(oss.str());
      }
    }
  };

  test(Tensor::tensorIds({s0, r0, s1, s2}), {true, true, false, false});
  test(Tensor::tensorIds({s0, s1, s2, r0}), {true, true, true, false});
  test(Tensor::tensorIds({r0, s1, s2, s0}), {true, true, true, false});
  test(Tensor::tensorIds({s1, r0, s2, s0}), {true, true, true, false});
}

void testEmptySlice0() {

  Graph g;
  auto a            = Tensor::variable(g, {10, 10});
  auto b            = a.slice({0, 0}, {10, 0}).closedAliasGate();
  auto c            = b.reverse(1).closedAliasGate();
  auto d            = b.dimShuffle({{1, 0}}).closedAliasGate();
  auto e            = d.reshape({5, 0}).closedAliasGate();
  auto f            = e.modify().closedAliasGate();
  auto g_           = b.modify().closedAliasGate();
  auto aliasGateIds = Tensor::opIds({b, c, d, e, f, g_});
  g.tryOpenings0(
      aliasGateIds, CheckParallelWriteable::Yes, AllowMultiGateAlias::No);
  std::cout << g << std::endl;
  for (auto id : aliasGateIds) {
    if (g.aliasGateIsClosed(id)) {
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
