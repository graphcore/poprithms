// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMPUTE_HOST_BASEOPERATORS_HPP
#define POPRITHMS_COMPUTE_HOST_BASEOPERATORS_HPP

#include "boolimpl.hpp"
#include "ieeehalf.hpp"

#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <set>
#include <type_traits>

#include <poprithms/compute/host/error.hpp>
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
template <> class Abs<BoolImpl> {
public:
  BoolImpl operator()(BoolImpl a) const { return a; }
  static std::string name() { return name_<BoolImpl>("Abs"); }
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

template <typename T> class Sqrt {
public:
  T operator()(T a) const { return static_cast<T>(std::sqrt(a)); }
  static std::string name() { return name_<T>("Sqrt"); }
};

template <typename T, class Enable = void> class Ceil {};

template <class T>
class Ceil<T, typename std::enable_if_t<std::is_floating_point_v<T>>> {
public:
  T operator()(T a) const { return std::ceil(a); }
  static std::string name() { return name_<T>("Ceil"); }
};

template <class T>
class Ceil<T,
           typename std::enable_if_t<std::is_integral_v<T> ||
                                     std::is_same<T, BoolImpl>::value>> {
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
class Floor<T,
            typename std::enable_if_t<std::is_integral_v<T> ||
                                      std::is_same<T, BoolImpl>::value>> {
public:
  T operator()(T a) const { return a; }
  static std::string name() { return name_<T>("Floor"); }
};

template <typename T> class Adder {
public:
  T operator()(T a, T b) const { return a + b; }
  static std::string name() { return name_<T>("Adder"); }
};

template <typename T> class Multiplier {
public:
  T operator()(T a, T b) const { return a * b; }
  static std::string name() { return name_<T>("Multiplier"); }
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

// numpy throws an error when bools are subtracted, we do the same.
template <> class Subtracter<BoolImpl> {
public:
  [[noreturn]] BoolImpl operator()(BoolImpl, BoolImpl) const {
    throw error("No Subtraction defined for BoolImpl");
  }
  static std::string name() { return name_<BoolImpl>("Subtracter"); }
};

template <> class Divider<BoolImpl> {
public:
  [[noreturn]] BoolImpl operator()(BoolImpl, BoolImpl) const {
    throw error("No Division defined for BoolImpl");
  }
  static std::string name() { return name_<BoolImpl>("Divider"); }
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
                                        std::is_same<T, BoolImpl>::value ||
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
template <> class Sqrt<BoolImpl> {
public:
  BoolImpl operator()(BoolImpl a) const { return a; }
  static std::string name() { return name_<BoolImpl>("Sqrt"); }
};

} // namespace host
} // namespace compute
} // namespace poprithms

#endif
