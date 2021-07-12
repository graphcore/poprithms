// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_POPRITHMS_NDARRAY_DTYPE_HPP
#define GUARD_POPRITHMS_NDARRAY_DTYPE_HPP

#include <sstream>
#include <string>
#include <vector>

#include <poprithms/util/typedinteger.hpp>

namespace poprithms {
namespace ndarray {

/** Enumeration of numerical data types **/
enum class DType {
  // IEEE floating point numbers
  Float16 = 0,
  Float32,
  Float64,

  // Signed integers
  Int8,
  Int16,
  Int32,
  Int64,

  // Unsigned integers
  Boolean,
  Unsigned8,
  Unsigned16,
  Unsigned32,
  Unsigned64,

  // Number of types
  N
};

using DTypes = std::vector<DType>;

std::ostream &operator<<(std::ostream &, DType);

/** The number of bytes in 1 element of data type \a t */
int nbytes(DType t);
uint64_t nbytes_u64(DType);

/** A lowercase string representation of data type \a t */
const std::string &lcase(DType t);

/** A PascalCase string representation of data type \a t */
const std::string &pcase(DType);

/** \return true iff data type \a t is fixed point. In other words, if \a t is
 *          an integer type and not a floating point type. */
bool isFixedPoint(DType t);

/** \return true iff data type \a t is non-negative. */
bool isNonNegative(DType t);

/** \return true iff data type \a t is non-negative and fixed point. In other
 *          words, iff \a t is an unsigned intger. */
bool isUnsignedFixedPoint(DType t);

/** Assert that the template parameter T corresponds to the function argument
 * \a t. This function is useful for error checking where strongly typed and
 * untyped functions 'meet'. */
template <typename T> void verify(DType t);

/** \return The data type corresponding to the template parameter T. For
 *          example, get<float> returns (run-time) DType::Float32 */
template <typename T> DType get();

/** A template class for mapping data types (DTypes) to C++ basic types
 * (float, etc), For example, "ToType<DType::Float32>::Type" is equivalent to
 * "float" during C++ compilation. */
template <DType> class ToType {};

template <> DType get<double>();
template <> class ToType<DType::Float64> {
public:
  using Type = double;
};

template <> DType get<float>();
template <> class ToType<DType::Float32> {
public:
  using Type = float;
};

// float16 must be defined separately, depending on the implementation or
// supporting library used. Interestingly, there is a proposal to include half
// in the C++ standard, expected to land in C++23. See for example:
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0192r4.html

template <> DType get<int8_t>();
template <> class ToType<DType::Int8> {
public:
  using Type = int8_t;
};

template <> DType get<int16_t>();
template <> class ToType<DType::Int16> {
public:
  using Type = int16_t;
};

template <> DType get<int32_t>();
template <> class ToType<DType::Int32> {
public:
  using Type = int32_t;
};

template <> DType get<int64_t>();
template <> class ToType<DType::Int64> {
public:
  using Type = int64_t;
};

template <> DType get<uint8_t>();
template <> class ToType<DType::Unsigned8> {
public:
  using Type = uint8_t;
};

template <> DType get<bool>();
template <> class ToType<DType::Boolean> {
public:
  using Type = bool;
};

template <> DType get<uint16_t>();
template <> class ToType<DType::Unsigned16> {
public:
  using Type = uint16_t;
};

template <> DType get<uint32_t>();
template <> class ToType<DType::Unsigned32> {
public:
  using Type = uint32_t;
};

template <> DType get<uint64_t>();
template <> class ToType<DType::Unsigned64> {
public:
  using Type = uint64_t;
};

template <typename T> const std::string &lcase() { return lcase(get<T>()); }
template <typename T> const std::string &pcase() { return pcase(get<T>()); }

} // namespace ndarray
} // namespace poprithms

#endif
