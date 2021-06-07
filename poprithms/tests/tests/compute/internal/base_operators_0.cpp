// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/ndarray/dtype.hpp>

// These header files are not part of the public/installed poprithms API. They
// are found found at compile time because we have explicitly set the path to
// a directory in a parent CMakeLists.txt file.
#include <compute/host/include/baseoperators.hpp>
#include <compute/host/include/ieeehalf.hpp>

namespace {

using namespace poprithms::compute::host;

template <typename T>
void confirmUnary(T in_, T out_, T expected_, const std::string &name) {
  if (std::isnan(out_) || !(out_ - expected_ == T(0))) {
    std::ostringstream oss;
    oss << "Error for the base unary operator " << name << ". In = " << in_
        << ", expected = " << expected_ << ", out = " << out_
        << ". This for type " << poprithms::ndarray::lcase<T>() << '.';
    throw error(oss.str());
  }
}

template <typename InType, typename OutType = InType>
void confirmBinary(InType in0_,
                   InType in1_,
                   OutType out_,
                   OutType expected_,
                   const std::string &name) {
  if (std::isnan(out_) || !(out_ - expected_ == OutType(0))) {
    std::ostringstream oss;
    oss << "Error for the base binary operator " << name << ". In0 = " << in0_
        << ", In1 = " << in1_ << ", expected = " << expected_
        << ", out = " << out_ << ". This for input type "
        << poprithms::ndarray::lcase<InType>() << '.';
    throw error(oss.str());
  }
}

} // namespace

void test0() {

  // Identity
  const uint64_t three_u64{3};
  confirmUnary(1.3, Identity<double>()(1.3), 1.3, "Identity");
  confirmUnary(1.3f, Identity<float>()(1.3f), 1.3f, "Identity");
  confirmUnary(
      three_u64, Identity<uint64_t>()(three_u64), three_u64, "Identity");

  // Abs
  confirmUnary(-1.3, Abs<double>()(-1.3), 1.3, "Abs");
  confirmUnary(1.3f, Abs<float>()(1.3f), 1.3f, "Abs");
  confirmUnary(int64_t(-3), Abs<int64_t>()(int64_t(-3)), int64_t(3), "Abs");

  // Sqrt
  confirmUnary(9.0, Sqrt<double>()(9.0), 3.0, "Sqrt");

  bool caught = false;
  try {
    Sqrt<int>()(5);
  } catch (const poprithms::error::error &) {
    caught = true;
  }
  if (!caught) {
    throw error("Failed to catch case of square-rooting an int");
  }

  // Modder
  confirmUnary(7.5, Modder<double>()(7.5, 3), 1.5, "Mod");
  confirmUnary(7, Modder<int>()(7, 3), 1, "Mod");
  confirmUnary(true, Modder<bool>()(true, true), false, "Mod");
  confirmUnary(6., Modder<double>()(6, 3), 0., "Mod");
  confirmUnary(6, Modder<int>()(6, 3), 0, "Mod");
  confirmUnary(-8.5, Modder<double>()(-8.5, 3), -2.5, "Mod");
  confirmUnary(-9.5, Modder<double>()(-9.5, 3), -0.5, "Mod");
  confirmUnary(-7, Modder<int>()(-7, 3), -1, "Mod");

  // Ceil
  confirmUnary(9.01, Ceil<double>()(9.01), 10.0, "Ceil");
  confirmUnary(9.0, Ceil<double>()(9.0), 9.0, "Ceil");
  confirmUnary(5, Ceil<int>()(5), 5, "Ceil");

  // Floor
  confirmUnary(9.01, Floor<double>()(9.01), 9.0, "Floor");
  confirmUnary(9.0, Floor<double>()(9.0), 9.0, "Floor");
  confirmUnary(5, Floor<int>()(5), 5, "Floor");

  // Adder
  confirmBinary(4, 7, Adder<int>()(4, 7), 11, "Adder");

  // Multiplier
  confirmBinary(4, 7, Multiplier<int>()(4, 7), 28, "Multiplier");

  // Divider
  confirmBinary(4, 7, Divider<int>()(4, 7), 0, "Divider");
  confirmBinary(-4., 8., Divider<double>()(-4., 8.), -0.5, "Divider");

  // Subtracter
  confirmBinary(4, 7, Subtracter<int>()(4, 7), -3, "Subtracter");
  confirmBinary(-4., 8., Subtracter<double>()(-4., 8.), -12., "Subtracter");

  // GreaterThan
  confirmBinary(4, 7, GreaterThan<int>()(4, 7), false, "GreaterThan");
  confirmBinary(
      -4., 8., GreaterThan<double>()(-4., 8.), false, "GreaterThan");
  confirmBinary(
      -4., -4., GreaterThan<double>()(-4., -4.), false, "GreaterThan");
  confirmBinary(8., -4., GreaterThan<double>()(8., -4.), true, "GreaterThan");

  // GreaterThanOrEqualTo
  confirmBinary(
      3, 3, GreaterThanOrEqualTo<int>()(3, 3), true, "GreaterThanOrEqualTo");

  // LessThan
  confirmBinary(3, 3, LessThan<int>()(3, 3), false, "LessThan");

  // LessThanOrEqualTo
  confirmBinary(
      4, 7, LessThanOrEqualTo<int>()(4, 7), true, "LessThanOrEqualTo");
  confirmBinary(-4.,
                8.,
                LessThanOrEqualTo<double>()(-4., 8.),
                true,
                "LessThanOrEqualTo");
  confirmBinary(-4.,
                -4.,
                LessThanOrEqualTo<double>()(-4., -4.),
                true,
                "LessThanOrEqualTo");
  confirmBinary(8.,
                -4.,
                LessThanOrEqualTo<double>()(8., -4.),
                false,
                "LessThanOrEqualTo");

  // EqualTo
  confirmBinary(
      1.56f, 1.56f, EqualTo<float>()(1.56f, 1.56f), true, "EqualTo");
  confirmBinary(1, 2, EqualTo<int>()(1, 2), false, "EqualTo");
}

void test_bool_0() {
  const auto t = true;
  const auto f = false;
  confirmUnary<bool>(t, Abs<bool>()(t), t, "Abs(true)");
  confirmUnary<bool>(f, Abs<bool>()(f), f, "Abs(false)");
  confirmUnary<bool>(f, Ceil<bool>()(f), f, "Ceil(false)");
  confirmUnary<bool>(t, Sqrt<bool>()(t), t, "Sqrt(true)");
}

int main() {
  test0();
  test_bool_0();
  return 0;
}
