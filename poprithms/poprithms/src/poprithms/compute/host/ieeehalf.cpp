// Copyright (c) 2015 Graphcore Ltd. All rights reserved.
//
// ** Copied verbatim from poplar (September 2020) **

#include "./include/ieeehalf.hpp"

#include <cstring>
#include <math.h>

/*
 * MACROs for manipulating the raw bit format of half-precision values
 */
#define HALF_MANT_SHIFT (0)
#define HALF_MANT_SIZE (10)
#define HALF_MANT_MASK ((1 << HALF_MANT_SIZE) - 1)
#define HALF_EXP_SHIFT HALF_MANT_SIZE
#define HALF_EXP_SIZE (5)
#define HALF_EXP_MASK ((1 << HALF_EXP_SIZE) - 1)
#define HALF_MAX_EXP HALF_EXP_MASK
#define HALF_SIGN_SHIFT (HALF_EXP_SHIFT + HALF_EXP_SIZE)
#define HALF_Q_SHIFT (HALF_EXP_SHIFT - 1)
#define HALF_BIAS (15)
#define HALF_EXP(v) (((v) >> HALF_EXP_SHIFT) & HALF_EXP_MASK)
#define HALF_MANT(v) (((v) >> HALF_MANT_SHIFT) & HALF_MANT_MASK)
#define HALF_SIGN(v) (((v) >> HALF_SIGN_SHIFT) & 1)
#define HALF_IS_NEG(v) (HALF_SIGN(v) != 0)
#define HALF_IS_ZERO(v) ((HALF_EXP(v) == 0) && (HALF_MANT(v) == 0))
#define HALF_IS_SUBNORM(v) ((HALF_EXP(v) == 0) && (HALF_MANT(v) != 0))
#define HALF_IS_INFINITY(v)                                                  \
  ((HALF_EXP(v) == HALF_MAX_EXP) && (HALF_MANT(v) == 0))
#define HALF_IS_NAN(v) ((HALF_EXP(v) == HALF_MAX_EXP) && (HALF_MANT(v) != 0))
#define HALF_IS_QNAN(v) (HALF_IS_NAN(v) && (((v >> HALF_Q_SHIFT) & 1) == 1))
#define HALF_IS_SNAN(v) (HALF_IS_NAN(v) && (((v >> HALF_Q_SHIFT) & 1) == 0))
#define HALF_INFINITY (HALF_MAX_EXP << HALF_EXP_SHIFT)

/*
 * MACROs for manipulating the raw bit format of single-precision values
 */
#define SINGLE_MANT_SHIFT (0)
#define SINGLE_MANT_SIZE (23)
#define SINGLE_MANT_MASK ((1 << SINGLE_MANT_SIZE) - 1)
#define SINGLE_EXP_SHIFT SINGLE_MANT_SIZE
#define SINGLE_EXP_SIZE (8)
#define SINGLE_EXP_MASK ((1 << SINGLE_EXP_SIZE) - 1)
#define SINGLE_MAX_EXP SINGLE_EXP_MASK
#define SINGLE_SIGN_SHIFT (SINGLE_EXP_SHIFT + SINGLE_EXP_SIZE)
#define SINGLE_Q_SHIFT (SINGLE_EXP_SHIFT - 1)
#define SINGLE_BIAS (127)
#define SINGLE_EXP(v) (((v) >> SINGLE_EXP_SHIFT) & SINGLE_EXP_MASK)
#define SINGLE_MANT(v) (((v) >> SINGLE_MANT_SHIFT) & SINGLE_MANT_MASK)
#define SINGLE_SIGN(v) (((v) >> SINGLE_SIGN_SHIFT) & 1)
#define SINGLE_IS_NEG(v) (SINGLE_SIGN(v) != 0)
#define SINGLE_IS_ZERO(v) ((SINGLE_EXP(v) == 0) && (SINGLE_MANT(v) == 0))
#define SINGLE_IS_SUBNORM(v) ((SINGLE_EXP(v) == 0) && (SINGLE_MANT(v) != 0))
#define SINGLE_IS_INFINITY(v)                                                \
  ((SINGLE_EXP(v) == SINGLE_MAX_EXP) && (SINGLE_MANT(v) == 0))
#define SINGLE_IS_NAN(v)                                                     \
  ((SINGLE_EXP(v) == SINGLE_MAX_EXP) && (SINGLE_MANT(v) != 0))
#define SINGLE_IS_QNAN(v)                                                    \
  (SINGLE_IS_NAN(v) && ((((v) >> SINGLE_Q_SHIFT) & 1) == 1))
#define SINGLE_IS_SNAN(v)                                                    \
  (SINGLE_IS_NAN(v) && ((((v) >> SINGLE_Q_SHIFT) & 1) == 0))
#define SINGLE_INFINITY (SINGLE_MAX_EXP << SINGLE_EXP_SHIFT)

namespace copied_from_poplar {

namespace {

/* A very naive implementation of half to single-precision float conversion.
   We could implement a more optimal routine if required but in its current
   form it should allow us to easily make tweaks if required.
*/
float toSingle(uint16_t ihalf) {

  bool neg         = HALF_IS_NEG(ihalf);
  uint32_t iresult = (neg ? (1U << SINGLE_SIGN_SHIFT) : 0);
  float result;

  if (HALF_IS_ZERO(ihalf)) {
    /* +- Zero
       - nothing more to do
     */

  } else if (HALF_IS_SUBNORM(ihalf)) {
    /* Subnormal values - represented as normalised values in single precision
     * format
     */
    uint32_t mant = HALF_MANT(ihalf) << (SINGLE_MANT_SIZE - HALF_MANT_SIZE);
    int exp       = 0;
    while ((mant & (1 << SINGLE_MANT_SIZE)) == 0) {
      exp -= 1;
      mant <<= 1;
    }

    mant &= ~(1 << SINGLE_MANT_SIZE);
    exp += (SINGLE_BIAS - HALF_BIAS + 1);

    iresult = HALF_SIGN(ihalf) << SINGLE_SIGN_SHIFT;
    iresult |= (mant << SINGLE_MANT_SHIFT);
    iresult |= (exp << SINGLE_EXP_SHIFT);

  } else if (HALF_IS_INFINITY(ihalf)) {
    /* +- Infinity
     */
    iresult |= SINGLE_INFINITY;

  } else if (HALF_IS_QNAN(ihalf)) {
    /* +- qNaN
     */
    iresult = SINGLE_INFINITY;
    iresult |= (1 << SINGLE_Q_SHIFT);

  } else if (HALF_IS_SNAN(ihalf)) {
    /* +- sNaN
     */
    iresult = SINGLE_INFINITY;

    /* Mantissa must be non-zero but top mantissa bit must be zero
     */
    iresult |= 1;

  } else {
    /* Normalised value
     */
    iresult = HALF_SIGN(ihalf) << SINGLE_SIGN_SHIFT;
    iresult |= (HALF_MANT(ihalf) << (SINGLE_MANT_SIZE - HALF_MANT_SIZE))
               << SINGLE_MANT_SHIFT;
    iresult |= (HALF_EXP(ihalf) + (SINGLE_BIAS - HALF_BIAS))
               << SINGLE_EXP_SHIFT;
  }

  std::memcpy(&result, &iresult, sizeof(result));

  return result;
}

/* A naive implementation of single to half-precision floating-point
   conversion.
   - Should be easy to tweak if required, since the different cases are
   explicitly handled separately,
*/
uint16_t toHalf(float value) {

  uint16_t result;
  uint32_t ivalue;
  int exp;

  std::memcpy(&ivalue, &value, sizeof(ivalue));

  result = SINGLE_SIGN(ivalue) << HALF_SIGN_SHIFT;
  exp    = SINGLE_EXP(ivalue) - SINGLE_BIAS;

  if (exp < -24) {
    /* Very small values map to +-0
       - nothing more to do.
     */

  } else if (exp < -14) {
    /* Small numbers map to denorms - will lose precision
     */

    /* Shift the exponent into the mantissa
     */
    int shift     = -exp - (HALF_BIAS);
    uint16_t mant = 1 << ((HALF_MANT_SIZE - 1) - shift);

    /* Combine with the original mantissa shifted into place
     */
    mant |= SINGLE_MANT(ivalue) >>
            ((SINGLE_MANT_SIZE - HALF_MANT_SIZE) + shift + 1);

    result |= (mant << HALF_MANT_SHIFT);

  } else if (exp <= 15) {
    /* Normal numbers - will lose precision
     */
    uint16_t mant =
        SINGLE_MANT(ivalue) >> (SINGLE_MANT_SIZE - HALF_MANT_SIZE);

    result |= ((exp + HALF_BIAS) << HALF_EXP_SHIFT);
    result |= (mant << HALF_MANT_SHIFT);

  } else if (exp < 128) {
    /* Large numbers map to infinity
     */
    result |= HALF_INFINITY;

  } else if (isnan(value)) {
    /* NaNs map to NaNs
     */
    uint16_t mant =
        SINGLE_MANT(ivalue) >> (SINGLE_MANT_SIZE - HALF_MANT_SIZE);

    if (SINGLE_IS_QNAN(ivalue)) {
      mant |= (1 << HALF_Q_SHIFT);
    } else {
      mant &= ~(1 << HALF_Q_SHIFT);

      if (mant == 0) {
        /* Ensure NaNs stay as NaNs (non-zero mantissa)
         */
        mant |= 1;
      }
    }

    result |= HALF_INFINITY;
    result |= mant << HALF_MANT_SHIFT;

  } else {
    /* Infinity maps to infinity
     */
    result |= HALF_INFINITY;
  }

  return result;
}

} // anonymous namespace

IeeeHalf::IeeeHalf(float value) { ihalf = toHalf(value); }

IeeeHalf IeeeHalf::fromBits(uint16_t bitPattern) {
  IeeeHalf h;
  h.ihalf = bitPattern;
  return h;
}

IeeeHalf::operator float() const { return toSingle(ihalf); }

IeeeHalf &IeeeHalf::operator+=(float other) {
  ihalf = toHalf(static_cast<float>(*this) + other);
  return *this;
}

IeeeHalf &IeeeHalf::operator-=(float other) {
  ihalf = toHalf(static_cast<float>(*this) - other);
  return *this;
}

IeeeHalf &IeeeHalf::operator*=(float other) {
  ihalf = toHalf(static_cast<float>(*this) * other);
  return *this;
}

IeeeHalf &IeeeHalf::operator/=(float other) {
  ihalf = toHalf(static_cast<float>(*this) / other);
  return *this;
}

bool IeeeHalf::operator<(float other) const {
  return static_cast<float>(*this) < other;
}

bool IeeeHalf::operator>(float other) const {
  return static_cast<float>(*this) > other;
}

bool IeeeHalf::operator<=(float other) const {
  return static_cast<float>(*this) <= other;
}

bool IeeeHalf::operator>=(float other) const {
  return static_cast<float>(*this) >= other;
}

bool IeeeHalf::operator==(IeeeHalf other) const {
  // This can be done fairly easily without converting to float.
  if (isNaN() || other.isNaN())
    return false;
  if (isZero() && other.isZero())
    return true;
  return ihalf == other.ihalf;
}

bool IeeeHalf::operator!=(IeeeHalf other) const { return !(*this == other); }

bool IeeeHalf::operator==(float other) const {
  return static_cast<float>(*this) == other;
}
bool IeeeHalf::operator!=(float other) const {
  return static_cast<float>(*this) != other;
}

IeeeHalf IeeeHalf::operator-() const {
  return IeeeHalf::fromBits(ihalf ^ (1 << HALF_SIGN_SHIFT));
}

IeeeHalf IeeeHalf::operator+(float other) {
  return static_cast<float>(*this) + other;
}

IeeeHalf IeeeHalf::operator-(float other) {
  return static_cast<float>(*this) - other;
}

IeeeHalf IeeeHalf::operator*(float other) {
  return static_cast<float>(*this) * other;
}

IeeeHalf IeeeHalf::operator/(float other) {
  return static_cast<float>(*this) / other;
}

bool IeeeHalf::isNaN() const { return HALF_IS_NAN(ihalf); }

bool IeeeHalf::isqNaN() const { return HALF_IS_QNAN(ihalf); }

bool IeeeHalf::issNaN() const { return HALF_IS_QNAN(ihalf); }

bool IeeeHalf::isInf() const { return HALF_IS_INFINITY(ihalf); }

bool IeeeHalf::isNorm() const { return !HALF_IS_SUBNORM(ihalf); }

bool IeeeHalf::isZero() const { return HALF_IS_ZERO(ihalf); }

} // namespace copied_from_poplar
