// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <poprithms/memory/chain/chain.hpp>
#include <poprithms/memory/chain/error.hpp>
#include <poprithms/memory/chain/settutil.hpp>
#include <testutil/memory/nest/randomregion.hpp>

namespace {

using namespace poprithms::memory::chain;
using namespace poprithms::compute::host;
using poprithms::ndarray::Strides;

// A class which implements the methods of NonNativeSettSampler's Helper
// template class, specialized for the host::Tensor class.
class HostHelper {
public:
  static Shape shape(const Tensor &t) { return t.shape(); }
  static Tensor slice(const Tensor &t, Dimension d, uint64_t l, uint64_t u) {
    return t.slice(d, l, u);
  }
  static Tensor reshape(const Tensor &t, const Shape &s) {
    return t.reshape(s);
  }
  static Tensor concat(const std::vector<Tensor> &ts, uint64_t d) {
    return Tensor::concat(ts, d);
  }
  static Tensor flatten(const Tensor &t) { return t.flatten(); }
};

void baseTest(const Tensor &inTensor, const Region &r) {

  NonNativeSettSampler a;

  // The Tensor we want to check is correct:
  const auto observed = a.settSample<Tensor, HostHelper>(inTensor, r);

  // This method is already well tested, we use it as the baseline:
  const auto expected = inTensor.gather(r.getOns());

  if (!observed.allEquivalent(expected)) {
    std::ostringstream oss;
    oss << "Comparing the method for settSampling with just "
        << "slice and reshape, in settutil, "
        << " to the gather method native to the "
        << "host::Tensor class. "
        << "Using the settSample template, the returned Tensor was \n"
        << observed << ", using the gather approach, the returned Tensor was "
        << expected << ". They should be the same. ";
    throw error(oss.str());
  }
}

// Some basic examples:
void test0() {
  Tensor inTensor = Tensor::arangeInt32(0, 3 * 5 * 7, 1).reshape({3, 5, 7});
  baseTest(inTensor, Region::fromBounds({3, 5, 7}, {1, 1, 1}, {2, 4, 6}));
  baseTest(inTensor, Region::fromBounds({3, 5, 7}, {1, 1, 1}, {2, 1, 6}));
  baseTest(inTensor, Region::fromBounds({3, 5, 7}, {1, 0, 1}, {2, 5, 6}));
  baseTest(inTensor, Region::fromStrides({3, 5, 7}, Strides({2, 2, 2})));
  baseTest(inTensor, Region::fromStrides({3, 5, 7}, Strides({2, 1, 3})));
  baseTest(inTensor, Region::fromStrides({3, 5, 7}, Strides({1, 10, 3})));
  baseTest(inTensor, Region::fromStripe({3, 5, 7}, 2, {2, 1, 2}));
}

// Pummel test the template with random regions.
void testRandom() {
  const Shape shape({17, 13});
  Tensor inTensor = Tensor::arangeInt32(0, shape.nelms(), 1).reshape(shape);
  for (uint64_t i = 0; i < 2048; ++i) {
    auto maxSettDepth = 2 + i % 3;
    auto seed         = 1011 + i;
    auto r =
        poprithms::memory::nest::getRandomRegion(shape, seed, maxSettDepth);
    baseTest(inTensor, r);
  }
}
} // namespace

int main() {
  test0();
  testRandom();
  return 0;
}
