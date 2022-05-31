// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_UNWIND_DYNAMICATTRACTIONS_HPP
#define POPRITHMS_MEMORY_UNWIND_DYNAMICATTRACTIONS_HPP

#include <map>
#include <sstream>
#include <tuple>
#include <vector>

#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/memory/unwind/graph.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

using common::multiout::InIndex;
using common::multiout::OpId;
using common::multiout::TensorId;

/**
 * Sets the priorities on the relative layouts of the tensors involved in
 * dynamic slice and dynamic update. Motivated by the 3 poplibs methods
 * for creating dynamic slice tensors, createSliceTensor,
 * createSliceableTensor, & createSliceableTensorFromSlice.
 * */
class DynamicAttractions {

public:
  static DynamicAttractions Default() { return DynamicAttractions{}; }

  /**
   * How important is it to set the sliceable (large) tensor's layout based on
   * the slice tensor's layout?
   * */
  double sliceToSliceable() const { return sliceToSliceable_; }

  double sliceableToSlice() const { return sliceableToSlice_; }

  double vanillaSliceable() const { return vanillaSliceable_; }

  DynamicAttractions &sliceToSliceable(double d) {
    sliceToSliceable_ = d;
    return *this;
  }

  DynamicAttractions &sliceableToSlice(double d) {
    sliceableToSlice_ = d;
    return *this;
  }

  DynamicAttractions &vanillaSliceable(double d) {
    vanillaSliceable_ = d;
    return *this;
  }

  bool operator==(const DynamicAttractions &rhs) const {
    return tup() == rhs.tup();
  }
  bool operator!=(const DynamicAttractions &rhs) const {
    return !operator==(rhs);
  }

  bool operator<(const DynamicAttractions &rhs) const {
    return tup() < rhs.tup();
  }

  std::tuple<double, double, double> tup() const {
    return std::tuple<double, double, double>{
        sliceToSliceable_, sliceableToSlice_, vanillaSliceable_};
  }

private:
  double sliceToSliceable_ = 100.;
  double sliceableToSlice_ = 100.;
  double vanillaSliceable_ = 50.;

  DynamicAttractions() = default;
};

class DynamicSources {
public:
  DynamicSources(const TensorId &vanillaSliceableSource,
                 const TensorId &fromSliceableSource,
                 const TensorId &fromSliceSource)
      : vanillaSliceableSource_(vanillaSliceableSource),
        fromSliceableSource_(fromSliceableSource),
        fromSliceSource_(fromSliceSource) {}

  TensorId vanillaSliceableSource() const { return vanillaSliceableSource_; }
  TensorId fromSliceableSource() const { return fromSliceableSource_; }
  TensorId fromSliceSource() const { return fromSliceSource_; }

private:
  TensorId vanillaSliceableSource_;
  TensorId fromSliceableSource_;
  TensorId fromSliceSource_;
};

/**
 * A utility method for Ops such as dynamicUpdate and dynamicSlice, where you
 * can either set the layout of the slice tensor from the layout of the
 * sliceable tensor using the poplibs API createSliceFromSliceable, or you can
 * set the layout of the sliceable tensor based on the layout of the slice,
 * using the poplibs API createSliceableFromSlice.
 *
 * Consider the dynamic update case:
 *
 *                       |
 *                       |
 *                       v
 * --> [toUpdate]     [updater]     [offset] <---
 *         |             |             |
 *         |             |             |
 *         +-------------+-------------+
 *                       |
 *                 dynamic_update
 *                       |
 *                   [updated] ---->
 *
 *  where the output, #updated, has the same layout as the input, #toUpdate.
 *
 *  This can be modelled with this method template method as:
 *
 *                                     |
 *                                     v
 * --> [toUpdate] -----------+     [updater]     [offset] <---
 *         |                 |         |
 *         v                 |         v
 *  SliceFromSliceable       |  SliceableFromSlice
 *         |                 |         |
 *         v                 |         v
 *  [fromSliceableSource]    |  [fromSliceSource]
 *                           |
 *                        Identity
 *                           |
 *                        [updated] ---->
 *
 *  ValuedPairs
 *  ============
 *  (toUpdate, fromSliceSource, sliceableFromSliceValue)
 *  (fromSliceableSource, updater, sliceFromSliceableValue)
 *
 *
 * */
template <class TensorCreatorInserter>
DynamicSources growDynamic(const TensorCreatorInserter &tcInserter,
                           poprithms::memory::unwind::Graph &g,
                           const DynamicAttractions &atts,
                           const TensorId &slice,
                           const TensorId &sliceable) {

  /**
   * creates 3 source:
   *   #vanillaSliceableSource
   *   #fromSliceSource
   *   #fromSliceableSource
   *
   *  and insert attractions ( ==== ) between them and a target (either #slice
   *  or #sliceable)
   *
   *
   *   fromSliceSource ==== slicebable ==== vanillaSliceableSource
   *     ^                       |                  ^
   *     |                       |                  |
   *     |                       v                  |
   *   slice  ====  fromSliceableSource            { }
   * */

  const Shape sliceShape     = g.shape(slice);
  const Shape sliceableShape = g.shape(sliceable);

  // attraction 1: between the sliceable tensor and a generic (vanilla)
  // layout for sliceable tensors
  //
  // For poplar backend, should use poplibs' createSliceableTensor.
  TensorId vanillaSliceableSource{
      g.barrier({}, {sliceableShape}, "vanilla_sliceable"), 0};
  g.insertValuedPair(
      vanillaSliceableSource, sliceable, atts.vanillaSliceable());
  tcInserter.insertVanillaCreator(vanillaSliceableSource);

  // attraction 2:
  //
  // For poplar backend, should use poplibs' createSliceTensor.
  TensorId fromSliceableSource{
      g.barrier({sliceable}, {sliceShape}, "slice_from_sliceable"), 0};
  g.insertValuedPair(fromSliceableSource, slice, atts.sliceableToSlice());
  tcInserter.insertSliceFromSliceableCreator(fromSliceableSource);

  // attraction 3:
  //
  // For poplar backend, should use poplibs' createSliceableTensorFromSlice.
  TensorId fromSliceSource{
      g.barrier({slice}, {sliceableShape}, "sliceable_from_slice"), 0};
  g.insertValuedPair(fromSliceSource, sliceable, atts.sliceToSliceable());
  tcInserter.insertSliceableFromSliceCreator(fromSliceSource);

  return {vanillaSliceableSource, fromSliceableSource, fromSliceSource};
}

} // namespace unwind
} // namespace memory
} // namespace poprithms

#endif
