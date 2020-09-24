// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
//
// Copied verbatim from poplar (September 2020) + a few lines at end.
// Unfortunately, Poplar does not expose this class in a public API, and it
// also seems easier to c&p this file here, rather than adding Poplar as a
// dependency of poprithms. git history suggests this file is very stable in
// Poplar, so negligible risk of diverging.
//

// [start ... copy and paste from poplar

/*********************************************************************
 * Half-precision floating-point abstraction for Colossus
 * Instruction Set Simulator
 *
 * Notes:
 *   Most operations are performed by:
 *     1 Converting the half-precision value to single-precision (float)
 *     2 Performing the operation at single-precision accuracy
 *     3 Converting the result back to half-precision
 *
 *********************************************************************/

#ifndef poplar_IeeeHalf_hpp
#define poplar_IeeeHalf_hpp

#include <stdint.h>

// copy and paste from poplar ... end]

#include <poprithms/ndarray/dtype.hpp>

namespace copied_from_poplar {

// [start ... copy and paste from poplar

/**
 * @brief A IeeeHalf. The half is stored in memory as an actual IeeeHalf
 * but most operations are performed by converting it to a `float`, performing
 * the operation and converting back again.
 */
class IeeeHalf {

public:
  /**
   * Uninitialised.
   */
  IeeeHalf() = default;

  /**
   * Initialise from a single-precision fp value
   */
  IeeeHalf(float value);

  /**
   * Initialise from a raw 16-bit pattern (conforming to IEEE 754-2008
   * binary16 format)
   */
  static IeeeHalf fromBits(uint16_t bitPattern);

  /**
   * Type-cast to single-precision
   */
  operator float() const;

  /**
   * Obtain half-precision bit-pattern
   * @returns raw 16-bit bit-pattern, as described by IEEE 754-2008
   */
  uint16_t bit16() const { return ihalf; }

  //  /**
  //   * Obtain half-precision bit-pattern
  //   * @returns raw zero-extended 16-bit bit-pattern, as described by
  //   * IEEE 754-2008
  //   */
  //  uint32_t bitz32() const { return (uint32_t)ihalf; }

  // Destructive operators
  // IeeeHalf &operator=(const IeeeHalf &other) = default;
  IeeeHalf &operator+=(float other);
  IeeeHalf &operator-=(float other);
  IeeeHalf &operator*=(float other);
  IeeeHalf &operator/=(float other);

  // Comparison operators.
  bool operator<(float other) const;
  bool operator>(float other) const;
  bool operator<=(float other) const;
  bool operator>=(float other) const;
  bool operator==(IeeeHalf other) const;
  bool operator!=(IeeeHalf other) const;

  // These are necessary to avoid ambiguous comparisons when doing
  // `IeeeHalf == float`, because we could either convert the first parameter
  // to a float, or the second to an IeeeHalf. To avoid the ambiguity we
  // provide functions that require no conversion (internally they convert
  // *this to a float and then compare). The same is true for double.
  bool operator==(float other) const;
  bool operator!=(float other) const;

  // Unary negation.
  IeeeHalf operator-() const;

  // Basic arithmetic operators.
  IeeeHalf operator+(float other);
  IeeeHalf operator-(float other);
  IeeeHalf operator*(float other);
  IeeeHalf operator/(float other);

  bool isqNaN() const;
  bool issNaN() const;
  bool isNaN() const;
  bool isInf() const;
  bool isNorm() const;
  bool isZero() const;

private:
  uint16_t ihalf;
};

// copy and paste from poplar ... end]

} // namespace copied_from_poplar

namespace poprithms {
namespace compute {
namespace host {
using IeeeHalf = copied_from_poplar::IeeeHalf;
}
} // namespace compute
} // namespace poprithms

namespace poprithms {
namespace ndarray {
template <> DType get<copied_from_poplar::IeeeHalf>();

} // namespace ndarray
} // namespace poprithms

#endif // poplar_IeeeHalf_hpp
