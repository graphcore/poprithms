// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMPUTE_HOST_BASEDATA_HPP
#define POPRITHMS_COMPUTE_HOST_BASEDATA_HPP
#include <algorithm>
#include <cstring>
#include <memory>
#include <random>

#include <compute/host/include/baseoperators.hpp>
#include <compute/host/include/ieeehalf.hpp>
#include <poprithms/compute/host/error.hpp>
#include <poprithms/compute/host/viewchange.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace compute {
namespace host {

class BaseData;
template <class T> class OriginData;
template <class T> class AllocData;
template <class T> class PointerData;
template <class T> class ViewData;
using ConstDataPtrs  = std::vector<const BaseData *>;
using BaseDataSP     = std::shared_ptr<BaseData>;
using AllocBooleanSP = std::shared_ptr<AllocData<bool>>;

/**
 * Abstract base class to represent a Tensor's underlying data values. This
 * class has no Shape, the values are represented as a 1-D row major
 * allocation.
 *
 * This class currently has 3 non-abstract children:
 *  - AllocData,
 *  - PointerData,
 *  - ElementViewData,
 *                            BaseData
 *                           /        \.
 *                     OriginData       ViewData
 *                     /      \         --------
 *               AllocData   PointerData
 *               ---------   -----------
 *
 * OriginData  : An abstract class which represents a contiguous block of
 *               row-major data values
 *
 * AllocData   : A class which contains and manages a contiguous buffer of
 *               row-major data
 *
 * PointerData : A class which containts a pointer to externally managed
 *               memory. The pointer is the address of the first element in a
 *               contiguous buffer of row-major data
 *
 * ViewData     : A class which represents a view into data which is
 *                contained in OriginData objects. Currently, every data
 *                element in a ViewData object is represented individually
 *                with an address into an OriginData object. The ViewData
 *                class can represent arbitrarily unstructured, complex views
 *                into OriginData.
 *
 * TODO(T27832) At some point we should specialize the ViewData class to
 * compactly store more structured views, for example we could have
 * StridedViewData vs ElementalViewData. StridedViewData would correspond to
 * the numpy.ndarray class.
 *
 * As with the public Tensor class, BaseData uses PyTorch '_' notation: If a
 * method contains the suffix '_', then the returned BaseData contains aliases
 * to the object on which the method was called. Conversely, if the method
 * does not have a suffix '_', it is guaranteed to not create any aliases.
 * */
class BaseData {

public:
  virtual ~BaseData() = default;

  /**
   * Reduction operators.
   * */
  virtual BaseDataSP reduceSum(const Shape &from, const Shape &to) const = 0;
  virtual BaseDataSP reduceProduct(const Shape &from,
                                   const Shape &to) const                = 0;
  virtual BaseDataSP reduceMin(const Shape &from, const Shape &to) const = 0;
  virtual BaseDataSP reduceMax(const Shape &from, const Shape &to) const = 0;

  /**
   * Binary operators.
   *
   * For all of binary methods, the single argument must be of the same size
   * and type as this BaseData's.
   * */
  virtual BaseDataSP add(const BaseData &) const      = 0;
  virtual BaseDataSP mul(const BaseData &) const      = 0;
  virtual BaseDataSP pow(const BaseData &) const      = 0;
  virtual BaseDataSP divide(const BaseData &) const   = 0;
  virtual BaseDataSP mod(const BaseData &) const      = 0;
  virtual BaseDataSP subtract(const BaseData &) const = 0;

  /**
   * Elementwise comparison.
   * */
  virtual AllocBooleanSP greaterThan(const BaseData &) const          = 0;
  virtual AllocBooleanSP lessThan(const BaseData &) const             = 0;
  virtual AllocBooleanSP greaterThanOrEqualTo(const BaseData &) const = 0;
  virtual AllocBooleanSP lessThanOrEqualTo(const BaseData &) const    = 0;
  virtual AllocBooleanSP equalTo(const BaseData &) const              = 0;

  /**
   * Binary modifiers.
   *
   * The argument must be of the same size and type as this BaseData.
   * */
  virtual void divide_(const BaseData &) const   = 0;
  virtual void mod_(const BaseData &) const      = 0;
  virtual void subtract_(const BaseData &) const = 0;
  virtual void add_(const BaseData &) const      = 0;
  virtual void mul_(const BaseData &) const      = 0;
  virtual void pow_(const BaseData &) const      = 0;

  /**
   * Unary operators.
   * */
  virtual BaseDataSP abs() const          = 0;
  virtual BaseDataSP sqrt() const         = 0;
  virtual BaseDataSP ceil() const         = 0;
  virtual BaseDataSP floor() const        = 0;
  virtual BaseDataSP clone() const        = 0;
  virtual BaseDataSP toOriginData() const = 0;

  /**
   * Unary modifiers.
   * */
  virtual void abs_() const   = 0;
  virtual void sqrt_() const  = 0;
  virtual void ceil_() const  = 0;
  virtual void floor_() const = 0;

  /**
   * Non-aliasing, view-changing operators.
   * */
  static BaseDataSP
  concat(const ConstDataPtrs &, const Shapes &, uint64_t axis);
  virtual BaseDataSP
  slice(const Shape &, const Lower &, const Upper &) const           = 0;
  virtual BaseDataSP slice(const Shape &,
                           const NormalizedSliceParams &) const      = 0;
  virtual BaseDataSP gather(const Shape &,
                            uint64_t dimension,
                            const std::vector<int64_t> &where) const = 0;
  virtual BaseDataSP
  gather(const Shape &,
         const std::vector<std::vector<int64_t>> &where) const = 0;
  virtual BaseDataSP
  scatterToZero(const Shape &inShape,
                const Shape &outShape,
                const std::vector<std::vector<int64_t>> &where) const = 0;

  virtual BaseDataSP expand(const Shape &from, const Shape &to) const     = 0;
  virtual BaseDataSP dimShuffle(const Shape &, const Permutation &) const = 0;
  virtual BaseDataSP
  reverse(const Shape &, const std::vector<uint64_t> &dimensions) const = 0;
  virtual BaseDataSP
  subSample(const Shape &, const std::vector<uint64_t> &strides) const = 0;

  /**
   * Aliasing, view-changing operators.
   * */
  static BaseDataSP
  concat_(const ConstDataPtrs &, const Shapes &, uint64_t axis);
  virtual BaseDataSP
  slice_(const Shape &, const Lower &, const Upper &) const           = 0;
  virtual BaseDataSP slice_(const Shape &,
                            const NormalizedSliceParams &) const      = 0;
  virtual BaseDataSP gather_(const Shape &,
                             uint64_t dimension,
                             const std::vector<int64_t> &where) const = 0;
  virtual BaseDataSP
  gather_(const Shape &,
          const std::vector<std::vector<int64_t>> &where) const        = 0;
  virtual BaseDataSP expand_(const Shape &from, const Shape &to) const = 0;
  virtual BaseDataSP dimShuffle_(const Shape &,
                                 const Permutation &) const            = 0;
  virtual BaseDataSP toViewData_() const                               = 0;
  virtual BaseDataSP
  reverse_(const Shape &, const std::vector<uint64_t> &dimensions) const = 0;
  virtual BaseDataSP
  subSample_(const Shape &, const std::vector<uint64_t> &strides) const = 0;

  /**
   * \return The number of elements in this BaseData.
   * */
  int64_t nelms_i64() const { return static_cast<int64_t>(nelms_u64()); }
  virtual uint64_t nelms_u64() const = 0;

  /**
   * Append a summary of this BaseData to \a ost
   *
   * This summary does not contain any numerical values.
   *
   * \see appendValues.
   * */
  virtual void append(std::ostream &ost) const = 0;

  /**
   * Append a numpy.ndarray style string to \a ost, for this Tensor,
   * arranged into Shape \a shape. The number of elements in this BaseData
   * must be the same as the number of elements in \a shape.
   *
   * \see append */
  virtual void appendValues(std::ostream &ost, const Shape &shape) const = 0;

  /**
   * \return true iff this is an OriginData.
   * */
  virtual bool isOriginData() const = 0;

  /**
   * \return true iff this is a ViewData.
   * */
  bool isViewData() const { return !isOriginData(); }

  /**
   * \return true iff there are aliases between any 2 elements of this
   *         BaseData.
   * */
  virtual bool containsAliases() const = 0;

  /**
   * \return The numerical type of this BaseData.
   * */
  virtual ndarray::DType dtype() const = 0;

  /** \return true iff all elements in this BaseData are 0. */
  virtual bool allZero() const = 0;

  /** \return true iff all elements in this BaseData are non-0. */
  virtual bool allNonZero() const = 0;

  /** The elements of this BaseData, cast to Float64. */
  virtual std::vector<double> getFloat64Vector() const = 0;

  /** Cast this BaseData to Float64. */
  virtual std::shared_ptr<AllocData<double>> toFloat64() const = 0;

  virtual std::vector<float> getFloat32Vector() const         = 0;
  virtual std::shared_ptr<AllocData<float>> toFloat32() const = 0;

  virtual std::vector<uint16_t> getFloat16Vector_u16() const     = 0;
  virtual std::shared_ptr<AllocData<IeeeHalf>> toFloat16() const = 0;

  virtual std::vector<int64_t> getInt64Vector() const         = 0;
  virtual std::shared_ptr<AllocData<int64_t>> toInt64() const = 0;

  virtual std::vector<uint64_t> getUnsigned64Vector() const         = 0;
  virtual std::shared_ptr<AllocData<uint64_t>> toUnsigned64() const = 0;

  virtual std::vector<int32_t> getInt32Vector() const         = 0;
  virtual std::shared_ptr<AllocData<int32_t>> toInt32() const = 0;

  virtual std::vector<uint32_t> getUnsigned32Vector() const         = 0;
  virtual std::shared_ptr<AllocData<uint32_t>> toUnsigned32() const = 0;

  virtual std::vector<int16_t> getInt16Vector() const         = 0;
  virtual std::shared_ptr<AllocData<int16_t>> toInt16() const = 0;

  virtual std::vector<uint16_t> getUnsigned16Vector() const         = 0;
  virtual std::shared_ptr<AllocData<uint16_t>> toUnsigned16() const = 0;

  virtual std::vector<int8_t> getInt8Vector() const         = 0;
  virtual std::shared_ptr<AllocData<int8_t>> toInt8() const = 0;

  virtual std::vector<uint8_t> getUnsigned8Vector() const         = 0;
  virtual std::shared_ptr<AllocData<uint8_t>> toUnsigned8() const = 0;

  virtual std::vector<bool> getBoolVector() const         = 0;
  virtual std::shared_ptr<AllocData<bool>> toBool() const = 0;

  virtual std::vector<char> getNativeCharVector() const = 0;

  static void assertSameTypes(const ConstDataPtrs &);
  static void assertForConcat(const ConstDataPtrs &, const Shapes &inShapes);
};

std::ostream &operator<<(std::ostream &ost, const BaseData &d);

} // namespace host
} // namespace compute
} // namespace poprithms

#endif
