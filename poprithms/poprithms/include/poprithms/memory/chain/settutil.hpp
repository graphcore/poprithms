// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_CHAIN_SETTUTIL_HPP
#define POPRITHMS_MEMORY_CHAIN_SETTUTIL_HPP
#include <numeric>
#include <sstream>

#include <poprithms/memory/chain/chain.hpp>
#include <poprithms/memory/chain/type.hpp>
#include <util/copybyclone_impl.hpp>

namespace poprithms {
namespace memory {
namespace chain {

/**
 * Class for performing a settSample in terms of slices and reshapes. This is
 * useful for Tensor classes which do not natively support settSample, such as
 * poplar::Tensor.
 * */
class NonNativeSettSampler {

public:
  /**
   * Sample \a inTensor in Region \a r
   * */
  template <typename T, typename Helper>
  T settSample(const T &inTensor, const Region &r) const {

    // The Helper class is used to get the expected behaviour out of the
    // Tensor class. See for example the test file, settutil_0.cpp where a
    // Helper class for the host::Tensor class is defined.
    const Helper h;

    const auto inShape = r.shape();
    const Shape outShape(r.nelms());

    // 1) flatten \a tIn,
    // 2) sample the flattened Tensor,
    // 3) reshape back to the correct rank.
    const auto flatSett     = r.flatten().sett(0);
    const auto flatInTensor = h.flatten(inTensor);
    const auto flatOut =
        settSampleFinalDimension<T, Helper>(flatInTensor, flatSett);
    const auto out = h.reshape(flatOut, outShape);
    return out;
  }

private:
  void assertNonZeroRank(uint64_t) const;
  void assertSubCalledRank(const Shape &, uint64_t) const;

  /**
   * Returns a Tensor of the same rank as \a t0, where the final dimension
   * has been sampled recursively with \a sett. All other dimensions are
   * unchanged. Example: if t0 is Shape (2,4)
   *  [[ 0 1 2 3 ]
   *   [ 4 5 6 7 ]]
   *
   * and sett is ((1,1,1)), then the returned Tensor is
   *  [[ 1 3 ]
   *   [ 5 7 ]].
   *
   * Recall that a Sett of ((1,1,1)) is on=1, off=1, and phase=1. Letting '1'
   * denote on and '.' denote off, this Sett looks like:
   *          .1.1.1.1.1
   */
  template <typename T, typename Helper>
  T settSampleFinalDimension(const T &t0, const nest::Sett &sett) const {

    const static Helper helper;

    const Shape inShape = helper.shape(t0);
    const auto inRank   = inShape.rank_u64();
    assertNonZeroRank(inRank);

    const auto nOnFinalDim = sett.n(0, inShape.finalDim());

    // If the sett has no Stripes, then there is no sampling to do in the
    // final dimension, the full Tensor should be returned.
    if (nOnFinalDim == inShape.finalDim()) {
      return t0;
    }

    // If the sett is empty, then return an empty Tensor of the correct Shape.
    if (nOnFinalDim == 0) {
      return helper.slice(t0, Dimension(inRank - 1), 0, 0);
    }

    const auto &current = sett.atDepth(0);

    // canonical case:
    // 0   < a0   < ... <  a1  < finalDimSize.
    // divided into 3 sections, which are concatenated.
    //
    //      a0                 a1
    //       |                  |
    // 11....11111.....11111....111
    // =======
    //       --------------------
    //                          ===
    //
    const auto a0 = current.firstStartNotBefore(0);
    const auto a1 = current.lastStartNotAfter(inShape.finalDim());

    // Note that at this point it is still possible that a1 < a0.

    std::vector<T> toConcat;

    // process [0, a0) which has the tail of an incomplete Stripe in it.
    if (a0 > current.off()) {
      const int64_t localEnd =
          std::min(inShape.finalDim(), a0 - current.off());
      const auto prefix = settSampleFinalDimension<T, Helper>(
          helper.slice(t0, Dimension(inRank - 1), 0, localEnd),
          sett.fromDepth(1).phaseShifted(a0 - current.period()));
      if (localEnd == inShape.finalDim()) {
        return prefix;
      }
      toConcat.push_back(prefix);
    }

    // the reshape trick for whole periods: a0 -> a1.
    if (a1 > a0) {

      // 1 slice out the complete Stripes from the middle (the  ---- in the
      // diagram above).
      const auto flatSlice = helper.slice(t0, Dimension(inRank - 1), a0, a1);

      // The total number of complete Stripes in -----
      int64_t H = (a1 - a0) / current.period();

      // Construct a Shape which is 1 rank greater than inShape. We divide the
      // final dimension of flatSlice into 2:
      auto newShape   = inShape.get();
      newShape.back() = H;
      newShape.push_back(current.period());
      const auto shapeUp = helper.reshape(flatSlice, newShape);

      // shapeUp no looks like
      // 11111.....
      // 11111.....
      // because we've reshaped it to have all the 0's on the right. We can
      // therefore slice out the 1's.
      const auto sliceOff =
          helper.slice(shapeUp, Dimension(inRank), 0, current.on_u64());

      // Recursive call! This should reduce sliceOff further, if sett has more
      // than just 1 Sett (in the diagram above, it has just 1 Sett).
      const auto subCalled =
          settSampleFinalDimension<T, Helper>(sliceOff, sett.fromDepth(1));

      // Reshape down to the original rank.
      const auto subCalledShape = helper.shape(subCalled);
      assertSubCalledRank(subCalledShape, inRank);
      auto outShape = inShape.get();
      outShape.back() =
          subCalledShape.dim(inRank) * subCalledShape.dim(inRank - 1);
      const auto subCalledDown = helper.reshape(subCalled, outShape);
      toConcat.push_back(subCalledDown);
    }

    // the tail: a1 -> end
    if (a1 >= a0 && a1 != inShape.finalDim()) {
      const auto postFix = settSampleFinalDimension<T, Helper>(
          helper.slice(t0,
                       Dimension(inRank - 1),
                       a1,
                       std::min(inShape.finalDim(), a1 + current.on())),
          sett.fromDepth(1));
      toConcat.push_back(postFix);
    }

    return helper.concat(toConcat, inRank - 1);
  }
};

} // namespace chain
} // namespace memory
} // namespace poprithms

#endif
