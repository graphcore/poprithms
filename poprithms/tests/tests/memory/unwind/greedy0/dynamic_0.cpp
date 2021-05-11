// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/unwind/error.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/hosttensorhelper.hpp>
#include <poprithms/memory/unwind/solution.hpp>

namespace {

using namespace poprithms::memory::unwind;

void dynamicSlice0() {

  Graph g;

  const auto sliceable           = g.sink({10, 5}, "sliceable");
  const auto dynamicSliceLikeOut = g.dynamicSliceLike(sliceable, {10}, 10);
  const auto slice               = dynamicSliceLikeOut.slice();
  g.setName(slice.opId(), "slice");
  const auto sliceableTarget  = dynamicSliceLikeOut.sliceableTarget();
  const auto downstreamTarget = g.source({10}, "slice target");
  g.insertValuedPair(slice, downstreamTarget, 20);

  auto s             = Solution(g);
  const auto barVals = HostTensorHelper::arangeBarriers(g);

  // Assert that the sliceable Tensor has its layout determined by the
  // sliced Tensor.
  HostTensorHelper::get(s, sliceable, barVals)
      .assertAllEquivalent(
          HostTensorHelper::get(s, sliceableTarget, barVals));

  // Assert that the slice Tensor has its layout determined by the
  // downstream target.
  HostTensorHelper::get(s, downstreamTarget, barVals)
      .assertAllEquivalent(HostTensorHelper::get(s, slice, barVals));
}

void dynamicUpdate0() {

  // clang-format off
  //
  //  OpId Name           OpType           InTensors Shape      Attractors
  //  ---- ----           ------           --------- -----      ----------
  //  0    toUpdate       Sink             ()        (10,20,30) ((op=2,v=10),(op=5,v=1))
  //  1    updater        Sink             ()        (20)       ((op=3,v=20))
  //  2                   SliceToSliceable ((op=1))  (10,20,30) ((op=0,v=10))
  //  3                   SliceableToSlice ((op=0))  (20)       ((op=1,v=20))
  //  4    updated        Identity         ((op=0))  (10,20,30) ()
  //  5    toUpdateSource Barrier          ()        (10,20,30) ((op=0,v=1))
  //  6    updaterSource  Barrier          ()        (20)       ()
  //  7    updatedSource  Barrier          ()        (10,20,30) ()
  //
  // clang-format on

  Graph g;

  const Shape toUpdateShape{10, 20, 30};
  const Shape updaterShape{20};

  const auto toUpdate = g.sink(toUpdateShape, "toUpdate");
  const auto updater  = g.sink(updaterShape, "updater");

  double sliceableFromSliceValue{1};
  double sliceFromSliceableValue{1};

  const auto dynamicUpdateLikeOut = g.dynamicUpdateLike(
      toUpdate, updater, sliceableFromSliceValue, sliceFromSliceableValue);

  const auto updated = dynamicUpdateLikeOut.updated();
  g.setName(updated.opId(), "updated");

  const auto toUpdateSource = g.source(toUpdateShape, "toUpdateSource");
  const auto updaterSource  = g.source(updaterShape, "updaterSource");
  const auto updatedSource  = g.source(toUpdateShape, "updatedSource");

  auto assertSource = [](const Paths &paths, const TensorId &expectedSource) {
    if (paths.size() != 1) {
      throw error("Expected just 1 Path in assertSource");
    }
    if (paths[0].src() != expectedSource) {
      std::ostringstream oss;
      oss << "Expected Source is " << expectedSource << ", not "
          << paths[0].src();
      throw error(oss.str());
    }
  };

  // layouts derived from toUpdate
  //
  //  source ===== toUpdate    updater
  //                  |           |
  //                  +-----+-----+
  //                        |
  //                  dynamic_update
  //                        |
  //                     updated
  //
  //
  {
    auto g0 = g;
    g0.insertValuedPair(toUpdateSource, toUpdate, 100);
    const auto soln0 = Solution(g0);
    assertSource(soln0.inwardsPaths(toUpdate), toUpdateSource);
    assertSource(soln0.inwardsPaths(updated), toUpdateSource);
    assertSource(soln0.inwardsPaths(updater),
                 dynamicUpdateLikeOut.updaterTarget());
  }

  // layouts derived from updater
  //
  //            toUpdate    updater ===== source
  //                |           |
  //                +-----+-----+
  //                      |
  //                dynamic_update
  //                      |
  //                   updated
  //
  //
  {
    auto g0 = g;
    g0.insertValuedPair(updaterSource, updater, 100);
    const auto soln0 = Solution(g0);
    assertSource(soln0.inwardsPaths(toUpdate),
                 dynamicUpdateLikeOut.toUpdateTarget());
    assertSource(soln0.inwardsPaths(updated),
                 dynamicUpdateLikeOut.toUpdateTarget());
    assertSource(soln0.inwardsPaths(updater), updaterSource);
  }

  // layouts derived from updated
  //
  //            toUpdate    updater
  //                |           |
  //                +-----+-----+
  //                      |
  //                dynamic_update
  //                      |
  //                   updated ======= source
  {
    auto g0 = g;
    g0.insertValuedPair(updatedSource, updated, 100);
    const auto soln0 = Solution(g0);
    assertSource(soln0.inwardsPaths(updated), updatedSource);
    assertSource(soln0.inwardsPaths(toUpdate), updatedSource);
    assertSource(soln0.inwardsPaths(updater),
                 dynamicUpdateLikeOut.updaterTarget());
  }
}

} // namespace

int main() {
  dynamicSlice0();
  dynamicUpdate0();
}
