// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <memory>
#include <sstream>

#include <testutil/memory/nest/randomregion.hpp>
#include <testutil/memory/unwind/fullstate.hpp>
#include <testutil/memory/unwind/graph.hpp>
#include <testutil/memory/unwind/op.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/unwind/sumlike.hpp>

namespace {

using namespace poprithms;
using poprithms::common::multiout::InIndex;
using poprithms::common::multiout::OpId;
using poprithms::common::multiout::TensorId;
using poprithms::memory::unwind::Path;
using poprithms::memory::unwind::SumAttractions;

void testTieBreaker0() {

  unwindtoy::Graph g;

  // A chain of adds and reduces.
  //
  // inputs of successively reduced shape:
  auto in0 = g.input({3, 3, 3, 3});
  auto in1 = g.input({1, 3, 3, 3});
  auto in2 = g.input({1, 1, 3, 3});
  auto in3 = g.input({1, 1, 1, 3});
  auto in4 = g.input({1, 1, 1, 1});

  auto r0 = g.reduce(in0, {1, 3, 3, 3});
  auto s0 = g.sum({r0, in1}, {0}, SumAttractions(10.));

  auto r1 = g.reduce(s0, {1, 1, 3, 3});
  auto s1 = g.sum({r1, in2}, {0}, SumAttractions(10.));

  auto r2 = g.reduce(s1, {1, 1, 1, 3});
  auto s2 = g.sum({r2, in3}, {0}, SumAttractions(10.));

  auto r3 = g.reduce(s2, {1, 1, 1, 1});
  g.sum({r3, in4}, {0}, SumAttractions(10.));

  unwindtoy::FullState fs(g);

  fs.lower();

  const auto ss = fs.scheduledSolution();

  // Assert that the i'th node in the schedule corresponds to the op #mlId.
  auto assertOp = [&ss](uint64_t i, OpId mlId) {
    // first, check that the i'th node is not a path:
    if (!ss.isOp(ss.schedule(i))) {
      std::ostringstream oss;
      oss << "Expected the schedule element # " << i << " to "
          << "be an op. ";
      throw poprithms::test::error(oss.str());
    }

    if (ss.op(ss.schedule(i)) != mlId) {
      std::ostringstream oss;
      oss << "The op at position " << i << " in the schedule is "
          << ss.op(ss.schedule(i)) << ", but " << mlId << " was expected. ";
      throw poprithms::test::error(oss.str());
    }
  };

  // Assert that the i'th node in the schedule is a path
  auto assertIsPath = [&ss](uint64_t i) {
    if (ss.isOp(ss.schedule(i))) {
      std::ostringstream oss;
      oss << "Expected the schedule element # " << i << " to "
          << "be a path. ";
      throw poprithms::test::error(oss.str());
    }
  };

  // Assert that the i'th node in the schedule is a path from #src to #dst
  auto assertPath =
      [&fs, &ss, &assertIsPath](uint64_t i, TensorId src, TensorId dst) {
        assertIsPath(i);

        const auto &p = ss.pathToSink(ss.schedule(i));

        if (p.src() != fs.toUnwind(src)) {
          std::ostringstream oss;
          oss << "Expected the source of the path of node # " << i
              << " to be " << src;
          throw poprithms::test::error(oss.str());
        }

        if (p.dst() != fs.toUnwind(dst)) {
          std::ostringstream oss;
          oss << "Expected the destination of the path of node # " << i
              << " to be " << dst;
          throw poprithms::test::error(oss.str());
        }
      };

  // Due to the longest path tie-breaker, we expect  only the first input to
  // have a linear (default) mapping, and all the other inputs to have layouts
  // set for being added to the reduction outputs.

  // Expect:
  // 0  Path to (op=0) : Source=(op=9),  Destination=(op=8),  Chain=()
  assertIsPath(0);

  // 1  Op : Input::0
  assertOp(1, in0.opId());
  // 2  Op : Reduce::5
  assertOp(2, r0.opId());
  // 3  Path to (op=1) : Source=(op=10),  Destination=(op=6),  Chain=()
  assertPath(3, r0, in1);
  // 4  Op : Input::1
  assertOp(4, in1.opId());
  // 5  Op : Sum::6
  assertOp(5, s0.opId());
  // 6  Op : Reduce::7
  assertOp(6, r1.opId());
  // 7  Path to (op=2) : Source=(op=12),  Destination=(op=4),  Chain=()
  assertPath(7, r1, in2);
  // 8  Op : Input::2
  assertOp(8, in2.opId());
  // 9  Op : Sum::8
  assertOp(9, s1.opId());
  // 10  Op : Reduce::9
  assertOp(10, r2.opId());
  // 11  Path to (op=3) : Source=(op=14),  Destination=(op=2),  Chain=()
  assertPath(11, r2, in3);
  // etc...
  // 12  Op : Input::3
  // 13  Op : Sum::10
  // 14  Op : Reduce::11
  // 15  Path to (op=4) : Source=(op=16),  Destination=(op=0),  Chain=()
  // 16  Op : Input::4
  // 17  Op : Sum::12
}

// a toy model of a training graph with a single matmul.
void testTieBreaker1() {

  // All the matmuls have the same attraction values. So, the tie-breaker
  // should use the matmul in the forward pass as it has the longest path to a
  // terminal op.
  unwindtoy::Graph g;
  auto in0   = g.input({3, 4});
  auto in1   = g.input({4, 5});
  auto act0  = g.matmul(in0, in1);
  auto red0  = g.reduce(act0, {3, 1});
  auto grad0 = g.expand(red0, {3, 5});
  g.matmul(grad0, g.dimShuffle(in1, {{1, 0}}));
  g.matmul(g.dimShuffle(in0, {{1, 0}}), grad0);

  unwindtoy::FullState fs(g);
  fs.lower();

  auto ss    = fs.scheduledSolution();
  auto sched = ss.schedule();
  uint64_t nPaths{0};
  for (uint64_t i = 0; i < ss.nNodes(); ++i) {
    if (ss.isOp(sched[i])) {
    } else {
      ++nPaths;
      const auto &p = ss.pathToSink(sched[i]);
      auto vps      = fs.uwGraph().valuedPartners(p.src());
      auto dst      = fs.toToy(p.dst());
      if (dst != in0 && dst != in1) {
        throw poprithms::test::error(
            "Expect the targets of the paths in this mock matmul training "
            "test to be the inputs to the fwd pass matmul. ");
      }

      bool isLocalSource{false};
      for (auto vp : vps) {
        if (vp.tensorId() == p.dst()) {
          isLocalSource = true;
        }
      }
      if (!isLocalSource) {
        throw poprithms::test::error(
            "Expect the source of the unwind path to be one of the sources "
            "of the inputs to the fwd pass matmul");
      }
    }
  }

  if (nPaths != 2) {
    throw poprithms::test::error(
        "Expected exactly 2 paths, 1 to each of the fwd pass matmul inputs.");
  }
}

} // namespace

int main() {
  testTieBreaker0();
  testTieBreaker1();
  return 0;
}
