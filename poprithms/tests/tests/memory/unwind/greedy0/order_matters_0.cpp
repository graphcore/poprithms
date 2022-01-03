// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/hosttensorhelper.hpp>
#include <poprithms/memory/unwind/solution.hpp>

namespace {
using namespace poprithms::memory::unwind;
using namespace poprithms;

void test0() {

  /**

              Sink (0)              Sink (3)
                |                      |
                |                      |
 Source (6)     |                      |
   >->->-> DimShuffle (1)          Reverse (4) <-<-<-<
                |                      |           Source (7)
                |                      |
                |                      |
             MaxPool (2) ----------- Add (5) ---->

   */

  // how many points if the output of the dimshuffle has the same layout as
  // source (6) ?
  double source6_attraction{1000.};

  // how many points if the output of the dimshuffle has the same layout as
  // source (7) ?
  double source7_attraction{1.};

  // how many points for getting the inputs to Add to have the same layout?
  double add_attraction{100.};

  Graph g;
  auto x0 = g.sink({16, 10});
  auto x1 = g.dimShuffle(x0, {{1, 0}});
  auto x2 = g.barrier({x1}, {{5, 8}});

  auto x3 = g.sink({5, 8});
  auto x4 = g.reverse(x3, Dimensions({1}));

  auto x5 = g.sumLike(TensorIds{{x2, 0}, x4}, 0, add_attraction);

  auto x6 = g.source({10, 16});
  g.insertValuedPair(x6, x1, source6_attraction);

  auto x7 = g.source({5, 8});
  g.insertValuedPair(x7, x4, source7_attraction);

  // 1) set sink (0) layout (as source (6) is most valued)
  // 2) run maxpool
  // 3) set sink (1) layout based on maxpool (add is more important than
  //    source (7)).
  Solution foo(g);

  {
    auto barVals = HostTensorHelper::arangeBarriers(g);
    auto x0_     = HostTensorHelper::get(foo, x0, barVals);
    auto x6_     = HostTensorHelper::get(foo, x6, barVals);
    x0_.assertAllEquivalent(x6_.dimShuffle({{1, 0}}));

    auto x3_ = HostTensorHelper::get(foo, x3, barVals);
    auto x2_ = HostTensorHelper::get(foo, {x2, 0}, barVals);
    x3_.assertAllEquivalent(x2_.reverse(1));

    auto paths = foo.barriersToSinks();
    if (paths.size() != 2) {
      throw poprithms::test::error("paths to the 2 sinks");
    }

    // order matters, as x0 required by maxpool required by x3.
    if (paths[0].dst() != x0) {
      throw poprithms::test::error("First path should be to x0");
    }

    if (paths[1].dst() != x3) {
      throw poprithms::test::error("Second path should be to x3");
    }
  }
}

void overlappingCandidates0() {

  Graph g;

  // [ a b c d ]
  //   =====       source0 (less valuable)
  //     ======    source1 (very valuable)

  auto x       = g.sink({4});
  auto x0      = g.slice(x, 0, 3);
  auto x1      = g.slice(x, 1, 4);
  auto source0 = g.source({3});
  auto source1 = g.source({3});
  g.insertValuedPair(x0, source0, 10.0);
  g.insertValuedPair(x1, source1, 20.0);

  Solution foo(g);
  auto barVals  = HostTensorHelper::arangeBarriers(g);
  auto observed = HostTensorHelper::get(foo, x, barVals);
  observed.slice({0}, {1}).assertAllEquivalent(
      barVals.at(source0).slice({0}, {1}));
  observed.slice({1}, {4}).assertAllEquivalent(barVals.at(source1));
}

void overlappingCandidates1() {

  Graph g;

  int64_t d0 = 3;
  int64_t d1 = 2;

  // 04       045         01
  // 52  ->   267  --+->  23
  // 67              |
  //                 +->  45
  //                      67

  auto x  = g.sink({d0, d1});
  auto y  = g.reshape(x, {d1, d0});
  auto z0 = g.slice(y, {0, 0}, {d1, d0 - 1});
  auto z1 = g.slice(y, {0, 1}, {d1, d0});

  auto source0 = g.source(g.shape(z0), "source0");
  auto source1 = g.source(g.shape(z1), "source1");

  g.insertValuedPair(z0, source0, 10);
  g.insertValuedPair(z1, source1, 11);

  Solution foo(g);

  auto barVals      = HostTensorHelper::arangeBarriers(g);
  auto observed     = HostTensorHelper::get(foo, x, barVals);
  auto observedFlat = observed.flatten().toInt32();
  observedFlat.assertAllEquivalent(
      compute::host::Tensor::int32({6}, {0, 4, 5, 2, 6, 7}));
}

} // namespace

int main() {
  test0();
  overlappingCandidates0();
  overlappingCandidates1();
  return 0;
}
