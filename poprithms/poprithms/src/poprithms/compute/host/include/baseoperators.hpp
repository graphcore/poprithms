// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMPUTE_HOST_BASEOPERATORS_HPP
#define POPRITHMS_COMPUTE_HOST_BASEOPERATORS_HPP

#include "ieeehalf.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <type_traits>

#include <compute/host/error.hpp>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/ndarray/shape.hpp>

namespace poprithms {
namespace compute {
namespace host {

template <typename T> std::string name_(const std::string &id) {
  return id + "___" + ndarray::pcase<T>();
}

/**
 * Classes for unary and binary maths operations on numerical types.
 *
 * Unary operators can be used as:
 *
 * <code>
 *   Abs<float> absOp;
 *   assert(absOp(-1.3f) - 1.3f == 0);
 * </code>
 *
 * Binary operators can be used as:
 *
 * <code>
 *   Divider<float> divOp;
 *   assert(divOp(11, 5) - 2 == 0);
 * </code>
 *
 * */

template <typename T> class Abs {
public:
  T operator()(T a) const { return a > T(0) ? a : -a; }
  static std::string name() { return name_<T>("Abs"); }
};

// abs(x) = x for all bools.
template <> class Abs<bool> {
public:
  bool operator()(bool a) const { return a; }
  static std::string name() { return name_<bool>("Abs"); }
};

template <typename T> class Identity {
public:
  T operator()(T a) const { return a; }
  static std::string name() { return name_<T>("Identity"); }
};

//
// Note on use of C++ keyword inline here: Used to allow multiple definitions.
//
// Specifically, from https://en.cppreference.com/w/cpp/language/inline:
//
// "an inline function or an inline variable (since C++17) may be defined
//     in a header file that is #include'd in multiple source files."
//
template <typename T> inline T fromDouble(double t) {
  return static_cast<T>(t);
}

template <typename T> class NoIntType {
public:
  NoIntType(std::string name) : name_(name) {}
  T operator()(T) const {
    throw error(
        "Failure in " + name_ + "<" + ndarray::pcase(ndarray::get<T>()) +
        ">, only floating point (fp) types allowed. Explicity cast to "
        "fp type.");
  }

private:
  std::string name_;
};

template <typename T, class Enable = void> class Sqrt {};
template <typename T>
class Sqrt<T,
           typename std::enable_if_t<std::is_floating_point_v<T> ||
                                     std::is_same<T, IeeeHalf>::value>> {
public:
  T operator()(T a) const { return static_cast<T>(std::sqrt(a)); }
  static std::string name() { return name_<T>("Sqrt"); }
};
template <typename T>
class Sqrt<T, typename std::enable_if_t<std::is_integral_v<T>>>
    : public NoIntType<T> {
public:
  Sqrt() : NoIntType<T>("Sqrt") {}
};

template <typename T, class Enable = void> class Cos {};
template <typename T>
class Cos<T,
          typename std::enable_if_t<std::is_floating_point_v<T> ||
                                    std::is_same<T, IeeeHalf>::value>> {
public:
  T operator()(T a) const { return static_cast<T>(std::cos(a)); }
  static std::string name() { return name_<T>("Cos"); }
};
template <typename T>
class Cos<T, typename std::enable_if_t<std::is_integral_v<T>>>
    : public NoIntType<T> {
public:
  Cos() : NoIntType<T>("Cos") {}
};

template <typename T, class Enable = void> class Sin {};
template <typename T>
class Sin<T,
          typename std::enable_if_t<std::is_floating_point_v<T> ||
                                    std::is_same<T, IeeeHalf>::value>> {
public:
  T operator()(T a) const { return static_cast<T>(std::sin(a)); }
  static std::string name() { return name_<T>("Sin"); }
};
template <typename T>
class Sin<T, typename std::enable_if_t<std::is_integral_v<T>>>
    : public NoIntType<T> {
public:
  Sin() : NoIntType<T>("Sin") {}
};

template <typename T, class Enable = void> class Log {};
template <typename T>
class Log<T,
          typename std::enable_if_t<std::is_floating_point_v<T> ||
                                    std::is_same<T, IeeeHalf>::value>> {
public:
  T operator()(T a) const { return static_cast<T>(std::log(a)); }
  static std::string name() { return name_<T>("Log"); }
};
template <typename T>
class Log<T, typename std::enable_if_t<std::is_integral_v<T>>>
    : public NoIntType<T> {
public:
  Log() : NoIntType<T>("Log") {}
};

template <typename T, class Enable = void> class Exp {};
template <typename T>
class Exp<T,
          typename std::enable_if_t<std::is_floating_point_v<T> ||
                                    std::is_same<T, IeeeHalf>::value>> {
public:
  T operator()(T a) const { return static_cast<T>(std::exp(a)); }
  static std::string name() { return name_<T>("Exp"); }
};
template <typename T>
class Exp<T, typename std::enable_if_t<std::is_integral_v<T>>>
    : public NoIntType<T> {
public:
  Exp() : NoIntType<T>("Exp") {}
};

template <typename T, class Enable = void> class Ceil {};

template <class T>
class Ceil<T, typename std::enable_if_t<std::is_floating_point_v<T>>> {
public:
  T operator()(T a) const { return std::ceil(a); }
  static std::string name() { return name_<T>("Ceil"); }
};

template <class T>
class Ceil<T, typename std::enable_if_t<std::is_integral_v<T>>> {
public:
  T operator()(T a) const { return a; }
  static std::string name() { return name_<T>("Ceil"); }
};

template <typename T, class Enable = void> class Floor {};

template <class T>
class Floor<T, typename std::enable_if_t<std::is_floating_point_v<T>>> {
public:
  T operator()(T a) const { return std::floor(a); }
  static std::string name() { return name_<T>("Floor"); }
};

template <class T>
class Floor<T, typename std::enable_if_t<std::is_integral_v<T>>> {
public:
  T operator()(T a) const { return a; }
  static std::string name() { return name_<T>("Floor"); }
};

template <typename T> class Reciprocal {
public:
  T operator()(T a) const { return T(1) / a; }
  static std::string name() { return name_<T>("Reciprocal"); }
};

template <typename T> class Adder {
public:
  T operator()(T a, T b) const { return a + b; }

  // x + 0 = x : 0 is the identity under addition.
  static T identity() { return T(0); }
  static std::string name() { return name_<T>("Adder"); }
};

template <typename T> class MinTaker {
public:
  T operator()(T a, T b) const { return std::min(a, b); }

  // min(x, +inf) = x : +inf is the identity under min.
  static T identity() { return std::numeric_limits<T>::max(); }
  static std::string name() { return name_<T>("MinTaker"); }
};

template <typename T> class MaxTaker {
public:
  T operator()(T a, T b) const { return std::max(a, b); }

  // Don't use numeric_limits<T>::min(), which is the machine epsilon for
  // floating point numbers!
  // max(x, -inf) = x : -inf is the identity under max.
  static T identity() { return std::numeric_limits<T>::lowest(); }
  static std::string name() { return name_<T>("MaxTaker"); }
};

template <typename T> class Multiplier {
public:
  T operator()(T a, T b) const { return a * b; }

  // x * 1 = x : 1 is the identity under multiplication.
  static T identity() { return T(1); }
  static std::string name() { return name_<T>("Multiplier"); }
};

template <typename T> class CopyFrom {
public:
  T operator()(T, T b) const { return b; }
  static std::string name() { return name_<T>("CopyFrom"); }
};

template <typename T> class Exponentiater {
public:
  T operator()(T a, T b) const { return static_cast<T>(std::pow(a, b)); }
  static std::string name() { return name_<T>("Exponentiater"); }
};

template <> class Exponentiater<bool> {
public:
  // Ignoring the case of pow(false, false).
  bool operator()(bool a, bool) const { return a; }
  static std::string name() { return name_<bool>("Exponentiater"); }
};

template <typename T> class Divider {
public:
  T operator()(T a, T b) const { return a / b; }
  static std::string name() { return name_<T>("Divider"); }
};

template <typename T> class Subtracter {
public:
  T operator()(T a, T b) const { return a - b; }
  static std::string name() { return name_<T>("Subtracter"); }
};

template <> class Adder<bool> {
public:
  static bool identity() { return false; }
  bool operator()(bool a, bool b) const { return a || b; }
  static std::string name() { return name_<bool>("Adder"); }
};

template <> class Multiplier<bool> {
public:
  static bool identity() { return true; }
  bool operator()(bool a, bool b) const { return a && b; }
  static std::string name() { return name_<bool>("Multiplier"); }
};

// numpy throws an error when bools are subtracted, we do the same.
template <> class Subtracter<bool> {
public:
  [[noreturn]] bool operator()(bool, bool) const {
    throw error("No Subtraction defined for bool");
  }
  static std::string name() { return name_<bool>("Subtracter"); }
};

template <> class Divider<bool> {
public:
  [[noreturn]] bool operator()(bool, bool) const {
    throw error("No Division defined for bool");
  }
  static std::string name() { return name_<bool>("Divider"); }
};

template <typename T> class GreaterThan {
public:
  bool operator()(T a, T b) const { return a > b; }
  static std::string name() { return name_<T>("GreaterThan"); }
};

template <typename T> class GreaterThanOrEqualTo {
public:
  bool operator()(T a, T b) const { return a >= b; }
  static std::string name() { return name_<T>("GreaterThanOrEqualTo"); }
};

template <typename T> class LessThan {
public:
  bool operator()(T a, T b) const { return a < b; }
  static std::string name() { return name_<T>("LessThan"); }
};

template <typename T> class LessThanOrEqualTo {
public:
  bool operator()(T a, T b) const { return a <= b; }
  static std::string name() { return name_<T>("LessThanOrEqualTo"); }
};

template <typename T, class Enable = void> class EqualTo {};

// Comparison of floats and doubles with operator== generates compiler
// warnings. We get around this by checking >= and <=.
template <class T>
class EqualTo<T, typename std::enable_if_t<std::is_floating_point_v<T>>> {
public:
  bool operator()(T a, T b) const { return a >= b && a <= b; }
  static std::string name() { return name_<T>("EqualTo"); }
};

template <class T>
class EqualTo<T,
              typename std::enable_if_t<std::is_integral_v<T> ||
                                        std::is_same<T, IeeeHalf>::value>> {

public:
  bool operator()(T a, T b) const { return a == b; }
  static std::string name() { return name_<T>("EqualTo"); }
};

template <class T>
class Floor<T, typename std::enable_if_t<std::is_same<T, IeeeHalf>::value>> {
public:
  T operator()(T a) const { return std::floor(static_cast<float>(a)); }
  static std::string name() { return name_<T>("Floor"); }
};

template <class T>
class Ceil<T, typename std::enable_if_t<std::is_same<T, IeeeHalf>::value>> {
public:
  T operator()(T a) const { return std::ceil(static_cast<float>(a)); }
  static std::string name() { return name_<T>("Ceil"); }
};

template <> inline IeeeHalf fromDouble(double t) {
  return static_cast<IeeeHalf>(static_cast<float>(t));
}

// sqrt(x) = x for all bools.
template <> class Sqrt<bool> {
public:
  bool operator()(bool a) const { return a; }
  static std::string name() { return name_<bool>("Sqrt"); }
};

template <> class Reciprocal<bool> {
public:
  [[noreturn]] bool operator()(bool) const {
    throw error("No Reciprocal defined for bool");
  }
  static std::string name() { return name_<bool>("Reciprocal"); }
};

template <typename T, class Enable = void> class Modder {};

template <class T>
class Modder<T,
             typename std::enable_if_t<std::is_floating_point_v<T> ||
                                       std::is_same<T, IeeeHalf>::value>> {
public:
  Modder() {}
  T operator()(T a, T b) const { return std::fmod(a, b); }
  static std::string name() { return name_<T>("Modder"); }
};

template <class T>
class Modder<T, typename std::enable_if_t<std::is_integral_v<T>>> {
public:
  T operator()(T a, T b) const { return a % b; }
  static std::string name() { return name_<T>("Modder"); }
};

} // namespace host
} // namespace compute
} // namespace poprithms

#endif
