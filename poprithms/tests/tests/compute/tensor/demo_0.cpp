// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>

namespace {

using namespace poprithms::compute::host;

void constructors() {

  //  Creating a Tensor:
  //      [[ 2.000000 3.000000 ]
  //       [ 4.000000 5.000000 ]]
  const Shape shape2x2{2, 2};
  std::vector<float> vDataF32{2., 3., 4., 5.};

  // copy from vector:
  const Tensor t0 = Tensor::float32(shape2x2, vDataF32);

  // copy from pointer:
  const Tensor t1 = Tensor::copyFloat32(shape2x2, vDataF32.data());

  // Store a pointer (memory management not done by Tensor):
  const Tensor t2 = Tensor::refFloat32(shape2x2, vDataF32.data());

  // move from vector:
  const Tensor t3 = Tensor::float32(shape2x2, {2., 3., 4., 5.});

  // Could construct using arange, then reshape2x2ape_
  const Tensor t4 = Tensor::arangeFloat32(2., 6., 1.).reshape_(shape2x2);

  // Creating a random 2x2 Tensor:
  //    [[ 0.655681 0.171965 ]
  //     [ 0.385741 0.112530 ]]
  const Tensor tRandom = Tensor::uniformFloat32(
      /*low = */ 0., /* upp = */ 1., shape2x2, /* seed = */ 1011);

  // Some tests:
  for (auto t : {t0, t1, t2, t3, t4}) {
    if (!t.allClose(t0, /*relTol = */ 0., /* absTol = */ 0.)) {
      std::ostringstream oss;
      oss << t << " and " << t0 << " ARE close.";
      throw poprithms::test::error(oss.str());
    }
    if (t.allClose(tRandom, 0., 0.)) {
      std::ostringstream oss;
      oss << t << " and " << tRandom << " are NOT close.";
      throw poprithms::test::error(oss.str());
    }
  }

  // TODO(jn): create tasks for
  //
  // - constructing from dtype: Tensor({shape2x2, {1,2,3,4}, DType::Int32);
  // - linspace
  // - ones
  //
}

void supportedTypes() {

  std::vector<Tensor> tensors;

  // Floating point numbers
  tensors.push_back(Tensor::float64(1.0));
  tensors.push_back(Tensor::float32(1.0f));
  tensors.push_back(Tensor::float16(1.0f));

  // Unsigned integers
  tensors.push_back(Tensor::unsigned64(1));
  tensors.push_back(Tensor::unsigned32(1));
  tensors.push_back(Tensor::unsigned16(1));
  tensors.push_back(Tensor::unsigned8(1));
  tensors.push_back(Tensor::boolean(true));

  // Signed integers
  tensors.push_back(Tensor::int64(1));
  tensors.push_back(Tensor::int32(1));
  tensors.push_back(Tensor::int16(1));
  tensors.push_back(Tensor::int8(1));

  // Some tests:
  for (auto x : tensors) {
    Tensor::int32(1).assertAllEquivalent(x.toInt32());
  }
}

void comparisonAndOverloads() {
  const auto tensor0 = Tensor::int32(Shape({2}), {1, 7});
  const auto tensor1 = Tensor::int32(Shape({2}), {4, 2});

  // Comparison, as per numpy
  const auto comp0 = tensor0 < tensor1;
  comp0.assertAllEquivalent(Tensor::boolean({2}, {true, false}));

  // Operators are overloaded, so addition is
  const auto sum = tensor0 + tensor1;
  sum.assertAllEquivalent(Tensor::int32({2}, {1 + 4, 7 + 2}));
}

void usesPytorchUnderscore() {

  const auto tensor = Tensor::int32(-1);

  // Absolute value, not inplace:
  const auto abs0 = tensor.abs();
  tensor.assertAllEquivalent(Tensor::int32(-1));

  // Absolute value, inplace:
  const auto abs1 = tensor.abs_();
  tensor.assertAllEquivalent(Tensor::int32(1));

  // Currently supported (10/10/2020) are:
  //
  // "view changing"
  // --------------
  // reshape          reshape_
  // flatten          flatten_
  // expand           expand_
  // slice            slice_
  // dimShuffle       dimShuffle_
  // concat           concat_
  //
  // "unary and binary numpy"
  // -----------------------
  // add (+)          add_
  // mul (*)          mul_
  // subtract (-)     subtract_
  // divide (/)       divide_
  // abs              abs_
  // ceil             ceil_
  // floor            floor_
  // sqrt             sqrt_
  //
  // TODO(jn) create tasks for:
  //
  // - reverse_, reverse
  // - subSample_, subSample
  // - matmul, conv (should we use OpenBLAS, or another library?)
  //
}

void poplarStyleAliasing() {

  const auto tensor = Tensor::int32(
      {3, 5}, {0, 1, 2, 3, 4, 5, -6, 7, -8, 9, 10, 11, 12, 13, 14});
  //                             ==     ==

  // [[ 0   1  2   3  4  ]
  //  [ 5  -6  7  -8  9  ]
  //  [ 10  11 12  13 14 ]]
  std::cout << tensor << std::endl;

  //
  // slice_ : -6, 7, -8, 9
  //
  // reshape_ : -6, 7
  //            -8, 9
  //
  // dimShuffle_: -6, -8
  //               7,  9
  //
  // slice_ : -6, -8
  //
  // abs_   : 6, 8
  const auto pos0 = tensor.slice_({1, 1}, {2, 5})
                        .reshape_({2, 2})
                        .dimShuffle_({{1, 0}})
                        .slice_({0, 0}, {1, 2})
                        .abs_();

  if (!(tensor < Tensor::int32(0)).allZero()) {
    throw poprithms::test::error(
        "Expected all values to be non-zero, after calling abs_");
  }
}

void canCheckForAliases() {

  const auto tensor = Tensor::int32(Shape({3}), {1, 2, 3});
  if (tensor.containsAliases()) {
    throw poprithms::test::error(
        "allocation {1,2,3} does not contain aliases");
  }

  // inplace (view-changing) concatenation:
  const auto concatted = concat_({tensor, tensor}, /* axis = */ 0);
  if (!concatted.containsAliases()) {
    throw poprithms::test::error(
        "self-concattenation (inplace) contains aliases");
  }
}

} // namespace

int main() {
  constructors();
  supportedTypes();
  comparisonAndOverloads();
  usesPytorchUnderscore();
  poplarStyleAliasing();
  canCheckForAliases();
  return 0;
}
