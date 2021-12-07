// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/ndarray/broadcastsetter.hpp>

namespace {
using namespace poprithms::ndarray;

/**
 * Test which mocks poplar. Instead of creating poplar tensors, and testing
 * the correctness of tile mappings propagated through a series of
 * view-changes, we create poprithms::compute::host::Tensors. This is much
 * faster, resulting is rapid unit tests.
 * */
class Helper {

private:
  using T = poprithms::compute::host::Tensor;

public:
  static uint64_t rank_u64(const T &t) { return t.rank_u64(); }
  static Shape shape(const T &t) { return t.shape(); }
  static T dimShuffle(const T &t, const Permutation &p) {
    return t.dimShuffle_(p);
  }
  static T prependOnesReshape(const T &t, uint64_t n) {
    return t.prependOnesReshape_(n);
  }
  static uint64_t numElements(const T &t) { return t.nelms_u64(); }

  static T flatten(const T &t, uint64_t i0, uint64_t i1) {
    return t.flatten_(i0, i1);
  }

  static T create(uint64_t dim, const T &t) {
    std::vector<int64_t> outShape_(t.rank_u64(), 1);
    outShape_[dim] = t.dim(dim);
    return t.reduceSum(Shape(outShape_)).flatten();
  }

  static void setDst(const T &creation, const T &target) {
    target.update_(creation.reshape(target.shape()));
  }
};

void test0() {

  using T = poprithms::compute::host::Tensor;
  Helper h;

  {

    // get the 'layout' of a tensor of shape {4,1,6,1} based on the 'large'
    // operand of shape {3,4,5,6,7}.
    const auto src = T::randomInt32(-100, 100, {3, 4, 5, 6, 7}, 1011);
    const auto dst = T::randomInt32(-100, 100, {4, 1, 6, 1}, 1012);
    BroadcastSetter::srcToDst<Helper, T>(src, dst, h);
    dst.flatten().assertAllEquivalent(
        src.reduceSum({1, 4, 1, 6, 1}).flatten());
  }

  {
    // get the 'layout' of a scalar based on a rank-6 tensor.
    const auto src = T::randomInt32(-10, 10, {4, 5, 2, 1, 3, 1}, 1011);
    const auto dst = T::randomInt32(-1, 1, {}, 13);
    BroadcastSetter::srcToDst<Helper, T>(src, dst, h);
    dst.assertAllEquivalent(src.reduceSum({}));
  }

  {
    // get the 'layout' of a scalar based on a scalar
    const auto src = T::randomInt32(-10, 10, {}, 1011);
    const auto dst = T::randomInt32(-1, 1, {}, 13);
    BroadcastSetter::srcToDst<Helper, T>(src, dst, h);
    dst.assertAllEquivalent(src.reduceSum({}));
  }

  {
    // get the 'layout' of a tensor of the same shape as the src.
    const auto src = T::randomInt32(-10, 10, {5, 7}, 1011);
    const auto dst = T::randomInt32(-1, 1, {5, 7}, 13);
    BroadcastSetter::srcToDst<Helper, T>(src, dst, h);
    dst.assertAllEquivalent(src);
  }

  {
    const auto src = T::randomInt32(-10, 10, {5, 7}, 1011);
    const auto dst = T::randomInt32(-1, 1, {7}, 13);
    BroadcastSetter::srcToDst<Helper, T>(src, dst, h);
    dst.assertAllEquivalent(src.reduceSum({1, 7}).squeeze_());
  }
}

} // namespace

int main() {

  test0();

  return 0;
}
