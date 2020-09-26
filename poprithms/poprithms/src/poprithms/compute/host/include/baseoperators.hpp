// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMPUTE_HOST_BASEOPERATORS_HPP
#define POPRITHMS_COMPUTE_HOST_BASEOPERATORS_HPP

#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <set>

#include <poprithms/compute/host/error.hpp>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/ndarray/shape.hpp>

namespace poprithms {
namespace compute {
namespace host {

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
  static std::string name() { return "Abs"; }
};

template <typename T> class Identity {
public:
  T operator()(T a) const { return a; }
  static std::string name() { return "Identity"; }
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

template <typename T> class Sqrt {
public:
  T operator()(T a) const { return static_cast<T>(std::sqrt(a)); }
  static std::string name() { return "Sqrt"; }
};

template <typename T, class Enable = void> class Ceil {};

template <class T>
class Ceil<T, typename std::enable_if_t<std::is_floating_point_v<T>>> {
public:
  T operator()(T a) const { return std::ceil(a); }
  static std::string name() { return "FloatingPointCeil"; }
};

template <class T>
class Ceil<T, typename std::enable_if_t<std::is_integral_v<T>>> {
public:
  T operator()(T a) const { return a; }
  static std::string name() { return "IntegerCeil"; }
};

template <typename T, class Enable = void> class Floor {};

template <class T>
class Floor<T, typename std::enable_if_t<std::is_floating_point_v<T>>> {
public:
  T operator()(T a) const { return std::floor(a); }
  static std::string name() { return "FloatingPointFloor"; }
};

template <class T>
class Floor<T, typename std::enable_if_t<std::is_integral_v<T>>> {
public:
  T operator()(T a) const { return a; }
  static std::string name() { return "FloatingPointFloor"; }
};

template <typename T> class Adder {
public:
  T operator()(T a, T b) const { return a + b; }
  static std::string name() { return "Adder"; }
};

template <typename T> class Multiplier {
public:
  T operator()(T a, T b) const { return a * b; }
  static std::string name() { return "Multiplier"; }
};

template <typename T> class Divider {
public:
  T operator()(T a, T b) const { return a / b; }
  static std::string name() { return "Divider"; }
};

template <typename T> class Subtracter {
public:
  T operator()(T a, T b) const { return a - b; }
  static std::string name() { return "Subtracter"; }
};

// Note that the retured type is T
template <typename T> class GreaterThan {
public:
  bool operator()(T a, T b) const { return a > b; }
  static std::string name() { return "GreaterThan"; }
};

template <typename T> class GreaterThanOrEqualTo {
public:
  bool operator()(T a, T b) const { return a >= b; }
  static std::string name() { return "GreaterThanOrEqualTo"; }
};

template <typename T> class LessThan {
public:
  bool operator()(T a, T b) const { return a < b; }
  static std::string name() { return "LessThan"; }
};

template <typename T> class LessThanOrEqualTo {
public:
  bool operator()(T a, T b) const { return a <= b; }
  static std::string name() { return "LessThanOrEqualTo"; }
};

template <typename T, class Enable = void> class EqualTo {};

template <class T>
class EqualTo<T, typename std::enable_if_t<std::is_floating_point_v<T>>> {
public:
  bool operator()(T a, T b) const { return a >= b && a <= b; }
  static std::string name() { return "FloatingPointEqualTo"; }
};

template <class T>
class EqualTo<T, typename std::enable_if_t<std::is_integral_v<T>>> {
public:
  bool operator()(T a, T b) const { return a == b; }
  static std::string name() { return "IntegralEqualTo"; }
};

} // namespace host
} // namespace compute
} // namespace poprithms

#endif
