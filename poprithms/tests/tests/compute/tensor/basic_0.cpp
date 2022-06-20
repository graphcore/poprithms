// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <numeric>

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>

namespace {
using namespace poprithms::compute::host;

void testZero() {
  const auto t = Tensor::boolean({2}, {true, false});
  if (t.allZero() || t.allNonZero()) {
    throw poprithms::test::error("t contains a mix of true and false");
  }
  const auto tTrue  = Tensor::boolean({2}, {true, true});
  const auto tFalse = Tensor::boolean({2}, {false, false});
  if (!tTrue.allNonZero() || !tFalse.allZero()) {
    throw poprithms::test::error("tTrue are all true, tFalse are all false");
  }

  const auto t0 = Tensor::float64({3}, {0., 0., 0.});
  if (t0.allNonZero() || !t0.allZero()) {
    throw poprithms::test::error("t0 is all zeros");
  }
}

void testAllClose() {

  // testing
  // absolute(a - b) <= (atol + rtol * absolute(b)).

  Tensor t0        = Tensor::float32({1}, {10.0});
  Tensor t1        = Tensor::float32({1}, {11.0});
  const auto atol0 = 0.2;
  const auto atol1 = 1.5;
  const auto rtol0 = 0.02;
  const auto rtol1 = 0.15;

  if (!t0.allClose(t1, rtol1, atol1)) {
    throw poprithms::test::error("should be close atol1 and rtol1");
  }

  if (!t0.allClose(t1, rtol0, atol1)) {
    throw poprithms::test::error("should be close with atol0 and rtol1");
  }

  if (!t0.allClose(t1, rtol1, atol0)) {
    throw poprithms::test::error("should be close with atol1 and rtol0");
  }

  if (t0.allClose(t1, rtol0, atol0)) {
    throw poprithms::test::error("shouldn't be close with atol0 and rtol0");
  }

  t0.assertAllEquivalent(t0);
}

void testIdenticalTo() {

  const auto t0 = Tensor::int32(1);
  const auto t1 = Tensor::int32(1);
  if (t0.identicalTo(t1) || !t0.identicalTo(t0)) {
    throw poprithms::test::error("Error in indenticalTo test");
  }
}

void testIsOrigin() {
  const auto t0 = Tensor::int32({2, 2}, {2, 3, 4, 5});
  auto t1       = t0.reshape_({4}).slice_({1}, {3});
  if (t0.implIsView() || t1.implIsOrigin()) {
    throw poprithms::test::error("Error in testIsOrigin");
  }
}

void testAtSlice0() {
  const auto t0 =
      Tensor::arangeInt32(0, 4, 1).reshape({4, 1, 1}).expand({4, 3, 2});
  t0.at(1).assertAllEquivalent(Tensor::int32(1).expand({3, 2}));
  t0.at(2).assertAllEquivalent(Tensor::int32(2).expand({3, 2}));

  // Inplace slice creates reference to sliced tensor
  t0.at_(1).zeroAll_();
  t0.at(1).assertAllEquivalent(Tensor::int32(0).expand({3, 2}));
}

void testAtSlice1() {
  const auto t0 =
      Tensor::arangeInt32(0, 4, 1).reshape({4, 1, 1}).expand({4, 3, 2});

  // Slice on non-negative integers only:
  {
    bool caught{false};
    try {
      t0.at_(Tensor::int32(-1));
    } catch (const poprithms::error::error &) {
      caught = true;
    }
    if (!caught) {
      throw poprithms::test::error(
          "Failed to catch error of slicing with at(.) on negative index");
    }
  }

  // Slice on scalars only:
  {
    bool caught{false};
    try {
      t0.at_(Tensor::unsigned32({0, 2, 3}, {}));
    } catch (const poprithms::error::error &) {
      caught = true;
    }
    if (!caught) {
      throw poprithms::test::error(
          "Failed to catch error when slicing with at(.) on non-scalar");
    }
  }
}

void testSlice0() {
  auto a = Tensor::arangeInt32(0, 2 * 3 * 4, 1).reshape({2, 3, 4});
  auto b = a.slice({1, 0, 0}, {2, 3, 1});
  auto c = a.slice(Dimensions({0, 2}), {1, 0}, {2, 1});
  b.assertAllEquivalent(c);
}

void testAccumulate0() {
  Tensors ts;
  for (uint64_t i = 0; i < 10; ++i) {
    ts.push_back(Tensor::unsigned64(i).expand({3, 2}));
  }
  auto out = Tensor::accumulate_(ts, CommutativeOp::Sum);
  out.assertAllEquivalent(ts[0]);
  out.assertAllEquivalent(Tensor::unsigned64(45).expand_({3, 2}));
}

void assertl2Norm(const Tensor &t, double expected) {
  if (t.l2norm() != expected) {
    std::ostringstream oss;
    oss << "Failed in test of l2 norm. For Tensor " << t << ", expected "
        << expected << " but observed " << t.l2norm();
    throw poprithms::test::error(oss.str());
  }
}

void testl2norm() {
  assertl2Norm(Tensor::float64({2}, {3.0, 4.0}), 5.0);
  assertl2Norm(Tensor::unsigned8({5}, {1, 1, 1, 2, 3}), 4.0);
  assertl2Norm(Tensor::boolean({6}, {true, true, false, true, true, false}),
               2);
}

void testAllClose1() {

  // a (b) does not dominate b (a).
  auto a = Tensor::float64({1, 3}, {1.0, 1.09, 1.08});
  auto b = Tensor::float64({3, 1}, {1.0, 0.91, 0.92});

  bool caught{false};
  try {
    a.allClose(b, 0.1, 0.0);
  } catch (poprithms::error::error &e) {
    std::string w = e.what();
    if (w.find("dominat") == std::string::npos) {
      throw poprithms::test::error("Expected an error about one tensor not "
                                   "dominating the other. Not + '" +
                                   w + "'");
    }
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error(
        "Failed to catch error of incompatible tensor comparison");
  }
}

void testScalarFromElement() {

  auto a = Tensor::int16({3, 2}, {10, 11, 12, 13, 14, 15});
  auto b = a.scalarFromElement(4);
  if (b.dtype() != DType::Int16) {
    throw poprithms::test::error("Tensor::scalarFromElement did not return a "
                                 "tensor of the same type as the input");
  }
  if (b.shape() != Shape{}) {
    throw poprithms::test::error(
        "Tensor::scalarFromElement did not return a scalar");
  }
  if (b.getInt16(0) != 14) {
    throw poprithms::test::error("Tensor::testScalarFromElement did not "
                                 "return a scalar of the correct value");
  }
}

void testAllValuesTheSame() {
  auto a = Tensor::float32({3}, {1, 1.001, 1});
  if (a.allValuesTheSame()) {
    throw poprithms::test::error("1.001 != 1, not all elements of a are the "
                                 "same: failure in 'allValuesTheSame'");
  }

  auto b = Tensor::int64({2, 2, 1, 2}, {3, 3, 3, 3, 3, 3, 3, 3});
  if (!b.allValuesTheSame()) {
    throw poprithms::test::error("All values of the tensor b have value '3': "
                                 "failure in 'allValuesTheSame'");
  }
}

void testImplicitCastError() {
  auto a = Tensor::float32(1);
  auto b = Tensor::float64(2);

  bool caught{false};
  try {
    a.add_(b);
  } catch (const poprithms::error::error &e) {

    auto x = std::string(e.what());
    if (x.find("implicit casting of op inputs is never performed") ==
        std::string::npos) {
      throw poprithms::test::error("The error message isn't as expected");
    }
    caught = true;
  }

  if (!caught) {
    throw poprithms::test::error("Failed to catch implicit cast attempt");
  }
}

void testOperatorNotEqual0() {

  //
  // 1 2         1       0 1
  // 3 4   !=    4  ->   1 0
  // 5 6         5       0 1

  auto a = Tensor::int32({3, 2}, {1, 2, 3, 4, 5, 6});
  auto b = Tensor::int32({3, 1}, {1, 4, 5});
  auto c = (a != b);
  c.assertAllEquivalent(Tensor::boolean({3, 2}, {0, 1, 1, 0, 0, 1}),
                        "See the mask diagram");

  bool caught{false};
  std::string errm{"sdfoisdfsodifhsd"};
  try {
    c.assertAllEquivalent(Tensor::boolean({3, 2}, {0, 0, 0, 1, 1, 1}), errm);
  } catch (const poprithms::error::error &e) {
    caught              = true;
    std::string message = e.what();
    if (message.find(errm) == std::string::npos) {
      std::cout << message << std::endl;
      throw poprithms::test::error("Failed to find context in error message");
    }
  }
  if (!caught) {
    throw poprithms::test::error("Failed to catch error");
  }
}

void testOperatorNotEqual1() {

  auto a = Tensor::float32(3.001);
  auto b = Tensor::float32({3}, {2.000, 3.0, 3.001});

  if ((a != b).toInt32().reduceSum().getInt32(0) != 2) {
    std::ostringstream oss;
    oss << a << b << (a != b) << ". Exactly 2 elements in b  are not 3.001";
    throw poprithms::test::error(oss.str());
  }
  if (((a != b).toInt16() + (a == b).toInt16()).reduceSum().getInt16(0) !=
      3) {
    throw poprithms::test::error(
        "all values should be 1 (and there are 3 values)");
  }
}

} // namespace

int main() {

  testZero();
  testAllClose();
  testIdenticalTo();
  testIsOrigin();
  testAtSlice0();
  testAtSlice1();
  testSlice0();
  testAccumulate0();
  testl2norm();
  testAllClose1();
  testScalarFromElement();
  testAllValuesTheSame();
  testImplicitCastError();
  testOperatorNotEqual0();
  testOperatorNotEqual1();

  return 0;
}
