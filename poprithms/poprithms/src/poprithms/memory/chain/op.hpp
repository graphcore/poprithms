// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_CHAIN_OP_HPP
#define POPRITHMS_MEMORY_CHAIN_OP_HPP
#include <ostream>
#include <variant>

#include <poprithms/memory/chain/disjointregionsmapper.hpp>
#include <poprithms/memory/chain/type.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/copybyclone.hpp>
#include <poprithms/util/permutation.hpp>

namespace poprithms {
namespace memory {
namespace chain {

using ndarray::Dimensions;
using ndarray::Shape;
using Lower     = ndarray::Shape::Lower;
using Upper     = ndarray::Shape::Upper;
using Stride    = ndarray::Stride;
using Dimension = ndarray::Dimension;
using nest::DisjointRegions;
using nest::Region;
using util::Permutation;

// C++ 17, so this cannot appear in a public header as poprithms is C++11 API.
using Variant = std::variant<Shape, Region, Permutation, Dimensions>;

/** An attribute of an Op. The attribute is one of
 *     Shape,
 *     Region,
 *     Permutation, and
 *     Dimensions.
 *
 * The use of std::variant ensures that
 * sizeof(Attr) =
 * max(sizeof(Shape), sizeof(Region), sizeof(Permutation), sizeof(Dimensions))
 * */
class Attr {
public:
  Attr(const Shape &s) : v(s) {}
  Attr(const Region &r) : v(r) {}
  Attr(const Permutation &p) : v(p) {}
  Attr(const Dimensions &d) : v(d) {}

  const Shape &shape() const { return std::get<Shape>(v); }
  const Region &region() const { return std::get<Region>(v); }
  const Permutation &permutation() const { return std::get<Permutation>(v); }
  const Dimensions &dimensions() const { return std::get<Dimensions>(v); }
  const Variant &var() const { return v; }

private:
  Variant v;
};

/** A view changing operation in a Chain. */
class Op {
public:
  Op(Type t, const Shape &o, const Attr &a)
      : type_(t), outShape_(o), attr_(a) {}
  Type type() const { return type_; }
  Shape outShape() const { return outShape_; }
  const Attr &attr() const { return attr_; }

  bool operator==(const Op &) const;
  bool operator!=(const Op &rhs) const { return !operator==(rhs); }

  // Letting [s] denote a Tensor of Shape s,
  //
  // convert
  //     [in0] -> (op0) -> [out0] -> (ds) -> [out1]
  // to
  //     [in0] -> (DimShuffle) -> [?] -> (op0's type) -> [out1].
  //
  // by changing op0 inplace to be of type DimShuffle, and ds inplace
  // to be op0's type. If the swap is not possible, then return false and
  // leave op0 and ds unchanged.
  //
  static bool bubbleDimShuffleBack(const Shape &in0, Op &op0, Op &ds);

  // see bubbleDimShuffleBack, same idea but for Expand instead of DimShuffle
  static bool bubbleExpandBack(const Shape &in0, Op &op0, Op &ex);

  // see bubbleDimShuffleBack, same idea but for Reduce instead of DimShuffle
  static bool bubbleReduceBack(const Shape &in0, Op &op0, Op &red);

  // see bubbleDimShuffleBack, same idea but for Reshape instead of DimShuffle
  static bool bubbleReshapeBack(const Shape &in0, Op &op0, Op &rsh);

  // see bubbleDimShuffleBack, same idea but for Reverse instead of DimShuffle
  static bool bubbleReverseBack(const Shape &in0, Op &op0, Op &rev);

  // see bubbleDimShuffleBack, same idea but for SettSample instead of
  // DimShuffle
  static bool bubbleSettSampleBack(const Shape &in0, Op &op0, Op &sample);

  // see bubbleDimShuffleBack, same idea but for SettFillInto instead of
  // DimShuffle
  static bool bubbleSettFillIntoBack(const Shape &in0, Op &op0, Op &sfill);

  Type type_;
  Shape outShape_;
  Attr attr_;
};

} // namespace chain
} // namespace memory
} // namespace poprithms

#endif
