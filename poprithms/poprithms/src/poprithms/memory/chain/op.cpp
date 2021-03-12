// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <numeric>
#include <sstream>
#include <variant>

#include <memory/chain/op.hpp>
#include <poprithms/memory/chain/error.hpp>

namespace poprithms {
namespace memory {
namespace chain {

namespace {

// Map a class in a std::variant to its index in the std::variant.

template <typename> struct tag {};

template <typename T, typename V> struct getIndex;

template <typename T, typename... Ts>
struct getIndex<T, std::variant<Ts...>>
    : std::integral_constant<size_t,
                             std::variant<tag<Ts>...>(tag<T>()).index()> {};

template <typename T> struct getVariantIndex : getIndex<T, Variant> {};

} // namespace

bool Op::operator==(const Op &rhs) const {
  if (type() != rhs.type()) {
    return false;
  }

  if (outShape() != rhs.outShape()) {
    return false;
  }

  switch (attr().var().index()) {
  case (getVariantIndex<Region>()): {
    return attr().region().equivalent(rhs.attr().region());
  }
  case (getVariantIndex<Shape>()): {
    return attr().shape() == rhs.attr().shape();
  }
  case (getVariantIndex<Permutation>()): {
    return attr().permutation() == rhs.attr().permutation();
  }
  case (getVariantIndex<Dimensions>()): {
    return attr().dimensions() == rhs.attr().dimensions();
  }
  }

  throw error("Exited switch in Chain::operator== without returning");
}

bool Op::bubbleDimShuffleBack(const Shape &inShape0, Op &op0, Op &op1) {

  if (op1.type() != Type::DimShuffle) {
    throw error("Calling bubbleDimShuffleBack with op1 of incorrect type");
  }

  const auto t0        = op0.type();
  const auto outShape0 = op0.outShape();

  const auto outShape1     = op1.outShape();
  const auto oldDimShuffle = op1;
  const auto p             = oldDimShuffle.attr().permutation();

  switch (t0) {

  case Type::Expand: {
    const auto p_ = p.get();
    // This can be solved in the case where the new dimensions are all
    // constrained to the start.
    std::vector<uint64_t> pNew(
        p_.cbegin() + outShape0.rank_u64() - inShape0.rank_u64(), p_.cend());

    if (std::any_of(pNew.cbegin(), pNew.cend(), [&inShape0](auto v) {
          return v >= inShape0.rank_u64();
        })) {
      return false;
    }
    op0 = {Type::DimShuffle,
           inShape0.dimShuffle(Permutation(pNew)),
           Permutation(pNew)};
    op1 = {Type::Expand, outShape1, outShape1};
    return true;
  }
  case Type::Reduce:
    return false;

  //
  // From
  //    DimShuffle -> Reshape,
  // To
  //    Reshape -> DimShuffle.
  //
  // Example 1:
  //    (2,3,5) -> reshape    -> (2,3,1,5)
  //            -> dimshuffle -> (5,1,3,2)
  //  becomes
  //    (2,3,5) -> dimshuffle -> (5,3,2)
  //            -> reshape    -> (5,1,3,2).
  //
  //
  // Example 2:
  //   (2,3,25) -> reshape             -> (6,5,5)
  //            -> dimShuffle((1,2,0)  -> (5,5,6)
  // becomes
  //   (2,3,25) -> dimShuffle(2,0,1) -> (25,2,3)
  //            -> reshape           -> (5,5,6)
  //
  //
  // Example 3:
  //   (2,3,35,11) -> reshape    -> (6,5,7,11)
  //               -> dimshuffle -> (11,6,5,7)
  // becomes
  //   (2,3,35,11) -> dimShuffle -> (11,2,3,35)
  //               -> reshape    -> (11,6,5,7)
  //
  //
  case Type::Reshape: {
    auto x = inShape0.moveDimShuffleFirst(outShape0, p);
    if (!x.first) {
      return false;
    }
    const auto permBack = x.second;
    op0 = {Type::DimShuffle, inShape0.dimShuffle(permBack), permBack};
    op1 = {Type::Reshape, outShape1, outShape1};
    return true;
  }

  // From
  //   DimShuffle -> Reverse,
  // To
  //   Reverse    -> DimShuffle,
  // where the axes of reversal change when it swaps positions with
  // DimShuffle.
  case Type::Reverse: {
    const auto oldReverse = op0;
    op0                   = oldDimShuffle;
    op1                   = {Type::Reverse,
           outShape1,
           Dimensions(p.mapForward(oldReverse.attr().dimensions().get()))
               .sorted()};
    return true;
  }

  // From
  //   DimShuffle -> "SettOp",
  // to
  //   "SettOp"   -> DimShuffle,
  // where the Region of the "SettOp" is dimShuffled when the 2 Ops swap
  // positions.
  //
  case Type::SettFillInto:
  case Type::SettSample: {
    const auto oldSettOp = op0;
    const auto r         = oldSettOp.attr().region().dimShuffle(p);
    op0                  = {Type::DimShuffle, inShape0.dimShuffle(p), p};
    op1                  = {t0, outShape1, r};
    return true;
  }

  default:
    throw error("Unhandled case in bubbleDimShuffleBack");
  }
}

// TODO(T33170) add logic for Reshape/Expand/Reduce. Also, all other cases
// which return false in the bubble methods below could be filled in.
bool Op::bubbleExpandBack(const Shape &inShape0, Op &op0, Op &op1) {
  if (op1.type() != Type::Expand) {
    throw error("Calling bubbleExpand with op1 of incorrect type");
  }
  (void)inShape0;
  const auto t0 = op0.type();
  switch (t0) {
  case Type::DimShuffle:
    return false;
  case Type::Reduce:
    return false;
  case Type::Reshape:
    return false;
  case Type::Reverse:
    return false;
  case Type::SettSample:
    return false;
  case Type::SettFillInto:
    return false;
  default:
    throw error("Unhandled case in bubbleExpandBack");
  }
}

bool Op::bubbleReduceBack(const Shape &inShape0, Op &op0, Op &op1) {
  (void)inShape0;
  if (op1.type() != Type::Reduce) {
    throw error("Calling bubbleReduce with op1 of incorrect type");
  }
  const auto t0 = op0.type();
  switch (t0) {
  case Type::DimShuffle:
    return false;
  case Type::Expand:
    return false;
  case Type::Reshape:
    return false;
  case Type::Reverse:
    return false;
  case Type::SettSample:
    return false;
  case Type::SettFillInto:
    return false;
  default:
    throw error("Unhandled case in bubbleReduceBack");
  }
}

bool Op::bubbleReshapeBack(const Shape &inShape0, Op &op0, Op &op1) {

  (void)inShape0;
  if (op1.type() != Type::Reshape) {
    throw error("Calling bubbleReshape with op1 of incorrect type");
  }
  const auto t0 = op0.type();
  switch (t0) {
  case Type::DimShuffle:
    return false;
  case Type::Reduce:
    return false;
  case Type::Reshape:
    return false;
  case Type::Reverse:
    return false;
  case Type::SettSample:
    return false;
  case Type::SettFillInto:
    return false;
  default:
    throw error("Unhandled case in bubbleReshapeBack");
  }
}

bool Op::bubbleReverseBack(const Shape &inShape0, Op &op0, Op &op1) {

  if (op1.type() != Type::Reverse) {
    throw error("Calling bubbleReverseBack with op1 of incorrect type");
  }

  const auto t0        = op0.type();
  const auto outShape0 = op0.outShape();

  const auto outShape1  = op1.outShape();
  const auto oldReverse = op1;

  switch (t0) {

  case Type::DimShuffle: {
    const auto oldDimShuffle = op0;
    const auto p             = op0.attr().permutation();
    op0                      = {Type::Reverse,
           inShape0,
           Dimensions(p.mapBackward(oldReverse.attr().dimensions().get()))};
    op1                      = oldDimShuffle;
    return true;
  }

  case Type::Expand:
    return false;
  case Type::Reduce:
    return false;
  case Type::Reshape:
    return false;

  // Go from
  //   Reverse  -> "SettOp",
  // to
  //   "SettOp" -> Reverse,
  // where the Region of the SettOp gets reversed when the 2 Ops swap
  // position.
  case Type::SettFillInto:
  case Type::SettSample: {
    const auto oldSettOp = op0;
    const auto r         = oldSettOp.attr().region();
    const auto dims      = oldReverse.attr().dimensions().get();
    op0                  = {Type::Reverse, inShape0, Dimensions(dims)};
    op1                  = {t0, outShape1, r.reverse(dims)};
    return true;
  }

  default:
    throw error("Unhandled case in bubbleExpandBack");
  }
}

bool Op::bubbleSettSampleBack(const Shape &inShape0, Op &op0, Op &op1) {
  if (op1.type() != Type::SettSample) {
    throw error("Calling bubbleSettSample with op1 of incorrect type");
  }
  (void)inShape0;
  const auto t0 = op0.type();
  switch (t0) {
  case Type::DimShuffle:
    return false;
  case Type::Expand:
    return false;
  case Type::Reduce:
    return false;
  case Type::Reshape:
    return false;
  case Type::Reverse:
    return false;
  case Type::SettFillInto:
    return false;
  default:
    throw error("Unhandled case in bubbleSettSampleBack");
  }
}

bool Op::bubbleSettFillIntoBack(const Shape &inShape0, Op &op0, Op &op1) {
  if (op1.type() != Type::SettFillInto) {
    throw error("Calling bubbleSettFillInto with op1 of incorrect type");
  }
  (void)inShape0;
  const auto t0 = op0.type();
  switch (t0) {
  case Type::DimShuffle:
    return false;
  case Type::Expand:
    return false;
  case Type::Reduce:
    return false;
  case Type::Reshape:
    return false;
  case Type::Reverse:
    return false;
  case Type::SettSample:
    return false;
  default:
    throw error("Unhandled case in bubbleSettFillIntoBack");
  }
}

} // namespace chain
} // namespace memory
} // namespace poprithms
