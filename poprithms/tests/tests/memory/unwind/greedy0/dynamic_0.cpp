// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/unwind/dynamicattractions.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/hosttensorhelper.hpp>
#include <poprithms/memory/unwind/solution.hpp>

namespace {

using namespace poprithms::memory::unwind;

class TensorCreatorInserter {
public:
  void insertVanillaCreator(const TensorId &) const {
    // for poplar backend, will call createSliceableTensor.
  }
  void insertSliceFromSliceableCreator(const TensorId &) const {
    // for poplar backend, will call createSliceTensor.
  }
  void insertSliceableFromSliceCreator(const TensorId &) const {
    // for poplar backend, will call createSliceableTensorFromSlice.
  }
};

void dynamicSlice0(double externalSliceSource     = 20,
                   double externalSliceableSource = 10,
                   double sliceToSliceable        = 100,
                   double vanillaSliceable        = 10,
                   double sliceableToSlice        = 50) {

  Graph g;

  const auto slice        = g.sink({4, 3}, "slice");
  const auto sliceSource0 = g.source({4, 3}, "sliceSource");
  g.insertValuedPair(slice, sliceSource0, externalSliceSource);

  const auto sliceable        = g.sink({6, 4}, "sliceable");
  const auto sliceableSource0 = g.source({6, 4}, "sliceableSource");
  g.insertValuedPair(sliceable, sliceableSource0, externalSliceableSource);

  const auto dynamicSources = growDynamic<TensorCreatorInserter>(
      {},
      g,
      DynamicAttractions::Default()
          .sliceableToSlice(sliceableToSlice)
          .sliceToSliceable(sliceToSliceable)
          .vanillaSliceable(vanillaSliceable),
      slice,
      sliceable);

  auto s             = Solution(g);
  const auto barVals = HostTensorHelper::arangeBarriers(g);

  // who gets a layout first? Check the 3 options:
  bool sliceableSetFirst{false};
  if (vanillaSliceable > externalSliceSource &&
      vanillaSliceable > externalSliceableSource) {
    HostTensorHelper::get(s, sliceable, barVals)
        .assertAllEquivalent(HostTensorHelper::get(
            s, dynamicSources.vanillaSliceableSource(), barVals));
    sliceableSetFirst = true;
  }

  else if (externalSliceSource > vanillaSliceable &&
           externalSliceSource > externalSliceableSource) {
    HostTensorHelper::get(s, slice, barVals)
        .assertAllEquivalent(HostTensorHelper::get(s, sliceSource0, barVals));
    sliceableSetFirst = false;
  }

  // externalSliceableSource is largest.
  else {
    HostTensorHelper::get(s, sliceable, barVals)
        .assertAllEquivalent(
            HostTensorHelper::get(s, sliceableSource0, barVals));
    sliceableSetFirst = true;
  }

  // if sliceable is set first, how is slice set?
  if (sliceableSetFirst) {
    // 2 options left for slice:
    if (externalSliceSource > sliceableToSlice) {
      HostTensorHelper::get(s, slice, barVals)
          .assertAllEquivalent(
              HostTensorHelper::get(s, sliceSource0, barVals));
    } else {
      HostTensorHelper::get(s, slice, barVals)
          .assertAllEquivalent(HostTensorHelper::get(
              s, dynamicSources.fromSliceableSource(), barVals));
    }
  }

  // if slice is set first, how is slice set?
  else {
    // 3 options left for sliceable:
    if (vanillaSliceable > externalSliceableSource &&
        vanillaSliceable > sliceToSliceable) {
      HostTensorHelper::get(s, sliceable, barVals)
          .assertAllEquivalent(HostTensorHelper::get(
              s, dynamicSources.vanillaSliceableSource(), barVals));
    } else if (externalSliceableSource > vanillaSliceable &&
               externalSliceableSource > sliceToSliceable) {
      HostTensorHelper::get(s, sliceable, barVals)
          .assertAllEquivalent(
              HostTensorHelper::get(s, sliceableSource0, barVals));
    } else {
      HostTensorHelper::get(s, sliceable, barVals)
          .assertAllEquivalent(HostTensorHelper::get(
              s, dynamicSources.fromSliceSource(), barVals));
    }
  }
}

void dynamicUpdate0() {

  Graph g;

  const Shape toUpdateShape{10, 20, 30};
  const Shape updaterShape{20};

  const auto toUpdate = g.sink(toUpdateShape, "toUpdate");
  const auto updater  = g.sink(updaterShape, "updater");

  double sliceableFromSliceValue{1};
  double sliceFromSliceableValue{1};
  double vanillaSliceable{0.01};

  const auto dynamicUpdateLikeOut = growDynamic<TensorCreatorInserter>(
      {},
      g,
      DynamicAttractions::Default()
          .sliceableToSlice(sliceFromSliceableValue)
          .sliceToSliceable(sliceableFromSliceValue)
          .vanillaSliceable(vanillaSliceable),
      updater,
      toUpdate);

  const auto updated = g.identity(toUpdate);

  const auto toUpdateSource = g.source(toUpdateShape, "toUpdateSource");
  const auto updaterSource  = g.source(updaterShape, "updaterSource");
  const auto updatedSource  = g.source(toUpdateShape, "updatedSource");

  auto assertSource = [](const Paths &paths, const TensorId &expectedSource) {
    if (paths.size() != 1) {
      throw poprithms::test::error("Expected just 1 Path in assertSource");
    }
    if (paths[0].src() != expectedSource) {
      std::ostringstream oss;
      oss << "Expected Source is " << expectedSource << ", not "
          << paths[0].src();
      throw poprithms::test::error(oss.str());
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
                 dynamicUpdateLikeOut.fromSliceableSource());
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
                 dynamicUpdateLikeOut.fromSliceSource());
    assertSource(soln0.inwardsPaths(updated),
                 dynamicUpdateLikeOut.fromSliceSource());
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
                 dynamicUpdateLikeOut.fromSliceableSource());
  }
}

} // namespace

int main() {

  // 120 tests, still fast (less than 8 [ms] on my laptop).
  std::vector<double> vals{1, 2, 3, 4, 5};
  for (auto p : poprithms::util::enumeratePermutations(5)) {
    poprithms::util::append(std::cout, p);
    std::cout << std::endl;
    dynamicSlice0(vals[p[0]], vals[p[1]], vals[p[2]], vals[p[3]], vals[p[4]]);
  }

  dynamicUpdate0();
}
