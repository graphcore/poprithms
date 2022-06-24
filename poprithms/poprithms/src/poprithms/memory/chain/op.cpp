// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <numeric>
#include <sstream>
#include <variant>

#include <memory/chain/error.hpp>
#include <memory/chain/op.hpp>

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

    // Using that Expands are guaranteed to be rank preserving:
  case Type::Expand: {
    const auto interShape = inShape0.dimShuffle(p);
    op0                   = {Type::DimShuffle, interShape, p};
    op1                   = {Type::Expand, outShape1, outShape1};
    return true;
  }
  case Type::Reduce: {
    return false;
  }

  //
  // From
  //    DimShuffle -> Reshape,
  // To
  //    Reshape -> DimShuffle.
  //
  // Example 1:
  //    (2,3,5) -> reshape    -> (2,3,1,5)
  //            -> dimShuffle -> (5,1,3,2)
  //  becomes
  //    (2,3,5) -> dimShuffle -> (5,3,2)
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
  //               -> dimShuffle -> (11,6,5,7)
  // becomes
  //   (2,3,35,11) -> dimShuffle -> (11,2,3,35)
  //               -> reshape    -> (11,6,5,7)
  //
  //
  case Type::Reshape: {
    auto x = inShape0.moveDimShuffleBeforeReshape(outShape0, p);
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

  const auto t0        = op0.type();
  const auto outShape0 = op0.outShape();
  const auto outShape1 = op1.outShape();
  const auto oldExpand = op1;
  (void)inShape0;

  switch (t0) {
  case Type::DimShuffle:
    return false;
  case Type::Reduce:
    return false;
  case Type::Reshape: {
    //
    // Can we replace Reshape(w)->Expand(x)
    //           with Expand(y)->Reshape(z) ?
    //
    // Example :    (4,5) -> reshape
    //           -> (1,4,1,5) -> expand
    //           -> (2,4,7,5).
    //
    // Cannot be permuted. We can never permute here if the reshape changes
    // the rank.
    //                 Reshape      Expand
    //                 -------      ------
    // Example : (4,3,1) -> (1,12,1) -> (1,12,12)   Can permute.
    //
    // Example : (4,3,1) -> (1,12,1) -> (13,12,11)  Can permute.
    //
    // Example : (2,5,7) -> (2,35,1) -> (2,35,6)    Cannot permute.
    //
    // Example : (4,1,2) -> (2,1,4) -> (2,7,4).     Cannot permute!
    //
    // Example : (4,3,2,1) -> (24,1) -> (24,7).     Cannot permute.
    //
    // Currently the implemetation rule is that you can permute the Expand
    // backwards if:
    //
    //   1) No rank change.
    //   2) Expansion dimensions are 1 before the reshape.
    //   3) No flow between the dimensions partitioned by the 1's.
    //

    // 1)
    if (inShape0.rank_u64() != outShape0.rank_u64()) {
      return false;
    }

    // 2)
    const auto expInds = outShape0.numpyIndicesToExpand(outShape1);
    for (auto i : expInds) {
      if (inShape0.dim(i) != 1) {
        return false;
      }
    }

    // (2,3,1,1,5,7,1,8) -> (3,1,2,1,7,5,1,8) -> (3,1,2,10,7,5,10,8)
    //        =     -              =     -              ==     --
    //
    //        ===== 35             ===== 35
    //              --- 8                --- 8
    // 3)
    //
    // We will check the products between all pairs in edges. We use the
    // fact that reshape preserves number of elements to skip the check of the
    // range [0, expInds[0]).
    //
    auto edges = expInds;
    edges.push_back(inShape0.rank_i64());

    for (uint64_t e = 0; e < edges.size() - 1; ++e) {
      if (inShape0.dimProduct(edges[e], edges[e + 1]) !=
          outShape0.dimProduct(edges[e], edges[e + 1])) {
        return false;
      }
    }

    // At this point, we've established that the permutation is valid.
    auto vInterShape = inShape0.get();
    for (auto ei : expInds) {
      vInterShape[ei] = outShape1.dim(ei);
    }
    Shape interShape(std::move(vInterShape));
    op0 = {Type::Expand, interShape, interShape};
    op1 = {Type::Reshape, outShape1, outShape1};
    return true;
  }

  case Type::Reverse: {
    op1 = {Type::Reverse, outShape1, op0.attr().dimensions()};
    op0 = oldExpand;
    return true;
  }

  case Type::SettSample: {
    //         inShape0                 outShape0         outShape1
    // example: (5,6,7) -> settSample -> (5,1,7) -> expand (5,3,7) No.
    // example: (5,1,7) -> settSample -> (5,1,2) -> expand (5,8,2) Yes.

    // If all the expansion indices have size 1 before the SettSample, then
    // the permutation is valid.

    auto expInds   = outShape0.numpyIndicesToExpand(outShape1);
    auto bubblable = std::all_of(
        expInds.cbegin(), expInds.cend(), [&inShape0](uint64_t i) {
          return inShape0.dim(i) == 1ll;
        });

    if (bubblable) {
      auto vExpandShape = inShape0.get();
      for (auto expInd : expInds) {
        vExpandShape[expInd] = outShape1.dim(expInd);
      }

      Shape expandShape(vExpandShape);
      Region r(expandShape, op0.attr().region().setts());
      op0 = {Type::Expand, expandShape, expandShape};
      op1 = {Type::SettSample, outShape1, r};
    }

    return bubblable;
  }

  case Type::SettFillInto: {
    // The permutation from SettFillInto(a) -> Expand(b) to
    //                      Expand(c) -> SettFillInto(d) is always valid.
    //
    // Example: (7,1) -> SettFillInto -> (10,1) -> Expand (10, 4) yes.
    //          (1,1) -> SettFillInto -> (1,2) -> Expand (100, 2) yes.

    const auto expInds = outShape0.numpyIndicesToExpand(outShape1);
    auto vExpandShape  = inShape0.get();
    for (auto expInd : expInds) {
      vExpandShape[expInd] = outShape1.dim(expInd);
    }
    Shape expandShape(vExpandShape);
    Region r(outShape1, op0.attr().region().setts());
    op0 = {Type::Expand, expandShape, expandShape};
    op1 = {Type::SettFillInto, outShape1, r};

    return true;
  }

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
  if (op1.type() != Type::Reshape) {
    throw error("Calling bubbleReshapeBack with op1 of incorrect type");
  }
  const auto outShape0 = op0.outShape();
  const auto outShape1 = op1.outShape();

  const auto t0 = op0.type();
  switch (t0) {
  case Type::DimShuffle:
    return false;
  case Type::Expand:
    return false;
  case Type::Reduce:
    return false;
  case Type::SettSample: {

    std::vector<uint64_t> sampleDims;
    for (uint64_t i = 0; i < inShape0.rank_u64(); ++i) {
      if (inShape0.dim(i) != outShape0.dim(i)) {
        // sampleMask[i] = true;
        sampleDims.push_back(i);
      }
    }

    //
    // Can you replace
    //      inShape0 -> settSample -> outShape0 -> reshape -> outShape1
    // with,
    //      inShape -> reshape -> X -> settSample -> Y ?
    //
    // The logic for settSample and slice is identical, we can ask the Shape
    // class if it is possible for slice:
    auto summary = inShape0.moveReshapeBeforeSlice(outShape0, outShape1);

    if (!std::get<0>(summary)) {
      // Not possible.
      return false;
    }

    // the shape of the output of the reshape, after it has been bubbled back
    // to before the settSample ('X' above).
    const Shape interShape = std::get<1>(summary);

    // the dimensions which are sliced *after* the reshape:
    const Dimensions finalSampleDims = std::get<2>(summary);

    Region permutedRegion = [&]() {
      if (interShape.rank_u64() == 0) {
        return Region::createFull({});
      }
      return op0.attr().region().sampleAtPermutedDims(
          interShape, Dimensions(sampleDims), finalSampleDims);
    }();

    op0 = {Type::Reshape, interShape, interShape};
    op1 = {Type::SettSample, outShape1, permutedRegion};

    return true;
  }

  case Type::Reverse:
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
    throw error("Unhandled case in bubbleReverseBack");
  }
}

bool Op::bubbleSettSampleBack(const Shape &inShape0, Op &op0, Op &op1) {

  (void)inShape0;
  if (op1.type() != Type::SettSample) {
    throw error("Calling bubbleSettSampleBack with op1 of incorrect type");
  }

  const auto outShape0 = op0.outShape();
  const auto outShape1 = op1.outShape();
  const auto t0        = op0.type();

  std::vector<uint64_t> sampleDims;
  for (uint64_t i = 0; i < outShape0.rank_u64(); ++i) {
    if (outShape1.dim(i) != outShape0.dim(i)) {
      sampleDims.push_back(i);
    }
  }

  switch (t0) {
  case Type::DimShuffle:
    return false;
  case Type::Expand:
    return false;
  case Type::Reduce:
    return false;
  case Type::Reshape: {

    auto summary = inShape0.moveSliceBeforeReshape(outShape0, outShape1);

    if (!std::get<0>(summary)) {
      // Not possible to permute the 2 ops.
      return false;
    }
    const Shape interShape     = std::get<1>(summary);
    const Dimensions finalDims = std::get<2>(summary);

    Region permutedRegion = [&]() {
      if (interShape.rank_u64() == 0) {
        return Region::createFull({});
      }
      return op1.attr().region().sampleAtPermutedDims(
          inShape0, Dimensions(sampleDims), finalDims);
    }();

    op0 = {Type::SettSample, interShape, permutedRegion};
    op1 = {Type::Reshape, interShape, outShape1};
    return true;
  }

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
