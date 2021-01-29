// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <poprithms/memory/chain/chain.hpp>
#include <poprithms/memory/chain/error.hpp>

namespace {

using namespace poprithms::memory::chain;
using namespace poprithms::compute::host;

void test0() {

  // Construct a Chain.
  Chain c({10, 10});
  c.reverse(Dimensions({0, 1}));
  c.slice({2, 4}, {9, 8});
  c.flatten();
  c.subSample(Stride(2), Dimension(0));

  // Construct an input Tensor.
  const auto t0 = Tensor::arangeInt32(0, 100, 1).reshape({10, 10});

  // Approach 1: Apply transformations JIT to the Tensor:
  const auto t1 = t0.reverse({0, 1})
                      .slice({2, 4}, {9, 8})
                      .flatten()
                      .subSample(Stride(2), Dimension(0));

  // Approach 2: "Compile" the chain, and apply it to the Tensor:
  const auto t2 = c.canonicalized().apply(t0);

  // Assert that the results are the same.
  t1.assertAllEquivalent(t2);
}

void test1() {
  // Same idea as test0, with a different set of ops covered.

  Chain c({7, 8});
  c.slice({1, 1}, {6, 7});
  c.reshape({6, 5});
  c.expand({2, 6, 5});
  c.reverse(Dimension(1));
  c.dimShuffle({{1, 2, 0}});
  c.flatten();
  c.subSample(Stride(2), Dimension(0));

  Tensor t0 = Tensor::uniformFloat32(-10, 10, {7, 8}, 1011);

  // JIT computation:
  Tensor t1 = t0.slice({1, 1}, {6, 7})
                  .reshape({6, 5})
                  .expand({2, 6, 5})
                  .reverse(1)
                  .dimShuffle({{1, 2, 0}})
                  .flatten()
                  .subSample(Stride(2), Dimension(0));

  // Compile and compute:
  auto t2 = c.canonicalized().apply(t0);

  t1.assertAllEquivalent(t2);

  // A meta-test: is allEquivalent doing what we expect? The Tensor should be
  // equivalent to a copy with a small perturbation.
  if (t1.allEquivalent(
          t2 + Tensor::uniformFloat32(-1e-6, +1e-6, t2.shape(), 1011))) {
    throw error("The perturbed Tensor should not compare equivalent to the "
                "unperturbed Tensor");
  }
}

} // namespace

int main() {
  test0();
  test1();
  return 0;
}
