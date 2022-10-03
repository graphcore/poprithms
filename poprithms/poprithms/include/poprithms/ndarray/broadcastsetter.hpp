// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_NDARRAY_SETBYBROADCAST_HPP
#define POPRITHMS_NDARRAY_SETBYBROADCAST_HPP

#include <poprithms/util/permutation.hpp>

namespace poprithms {
namespace ndarray {

using poprithms::util::Permutation;

/**
 * This class helps to set a tensor dst based on a tensor src, where src
 * and dst are tensors which can be combined with numpy broadcasting rules,
 * and src 'dominates' dst.
 *
 * Running example:
 *   src: (3,4,5,6,7)
 *   dst:   (1,5,1,7)
 *
 * Task:
 *   set the layout of dst in preparation for performing a broadcast
 *   elementwise operation between src (whose layout is fixed) and dst.
 *
 *
 * The rules to set dst are based on the poplibs API,
 * 'createBroadcastOperator'. Given a tensor t and a single non-broadcast
 * dimension d, a 1-d Tensor of size t.dim(d) is created. In our example,
 * there are 2 dimensions which are non-broadcast (dimensions 2 and 4 of
 * src), and so we need to apply some view-changing dimShuffles and reshapes
 * before we can set 'dst'. Further explanantion in the method, srcToHost.
 *
 * For a particular Tensor class, a Helper class is required which performs
 * basic methods on a Tensor.
 * */

class BroadcastSetter {

public:
  template <class Helper, typename Tensor>

  static void srcToDst(Tensor src, Tensor dst, const Helper &h) {

    // Helper::shape must return the Shape of the Tensor argument.
    Helper::shape(src).assertNumpyDominates(Helper::shape(dst));

    // running example: 5 - 4 = 1.
    const auto deficit = Helper::rank_u64(src) - Helper::rank_u64(dst);

    // running example: (1,1,5,1,7)
    const auto prepaddedDst = Helper::prependOnesReshape(dst, deficit);

    // running example: (0,1,3,2,5). all of the ones to the start.
    const auto p0 =
        Permutation::toStartWithOnes(Helper::shape(prepaddedDst).get_u64());

    // running example: 3.
    const auto nOnes = Helper::shape(prepaddedDst).nDimsOfSize(1);

    // running example: (3,4,6,5,7)
    const auto shuffledSrc = Helper::dimShuffle(src, p0);

    // running example: (1,1,1,5,7)
    const auto shuffledPrepaddedDst = Helper::dimShuffle(prepaddedDst, p0);

    // running example : 5
    const auto r0 = Helper::rank_u64(src);

    // running example : (3,4,6,35)
    const auto flattenedSrc = Helper::flatten(shuffledSrc, nOnes, r0);

    // running example : create a tensor with 35 elements.
    const auto creation = h.create(nOnes, flattenedSrc);

    assertSumeNumElms(Helper::numElements(creation),
                      Helper::numElements(dst));

    h.setDst(creation, shuffledPrepaddedDst);
  }

private:
  static void assertSumeNumElms(uint64_t creation, uint64_t dst);
};

} // namespace ndarray
} // namespace poprithms
#endif
