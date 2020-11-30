// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMPUTE_HOST_ROWMAJOR_HPP
#define POPRITHMS_COMPUTE_HOST_ROWMAJOR_HPP
#include "ieeehalf.hpp"
#include "typeddata.hpp"

#include <cstring>
#include <memory>

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/compute/host/viewchange.hpp>
#include <poprithms/ndarray/dtype.hpp>

namespace poprithms {
namespace compute {
namespace host {

/**
 * A class to help the OriginData class with functionality which is not
 * template-parameter specific.
 */
class OriginDataHelper {
public:
  static std::vector<int64_t> getIota_i64(uint64_t N);
  static std::vector<uint16_t> float16ToUint16(const std::vector<IeeeHalf> &);
  static void
  assertSameBinaryOpNelms(uint64_t n0, uint64_t n1, const BaseData &td);
};

template <typename From, typename To>
std::vector<To> castPtrToVector(const From *from, uint64_t nElms) {
  std::vector<To> r(nElms);
  std::transform(from,
                 std::next(from, static_cast<int64_t>(nElms)),
                 r.begin(),
                 [](From v) { return static_cast<To>(v); });
  return r;
}

template <>
std::vector<IeeeHalf> castPtrToVector<double, IeeeHalf>(const double *from,
                                                        uint64_t nElms);

/**
 * A BaseData with contiguous, row-major elements.
 * */
template <class T>
class OriginData : public TypedData<T>,
                   public std::enable_shared_from_this<OriginData<T>> {

public:
  using Vec = std::vector<T>;
  using BaseData::nelms_i64;
  using BaseData::nelms_u64;

  virtual T *dataPtr() const = 0;
  bool isOriginData() const final { return true; }
  bool containsAliases() const final { return false; }

  std::vector<char> getNativeCharVector() const final {
    std::vector<char> chars(nelms_u64() * sizeof(T));
    std::memcpy(chars.data(), dataPtr(), chars.size());
    return chars;
  }

  BaseDataSP expand(const Shape &from, const Shape &to) const final {
    return std::make_shared<AllocData<T>>(
        ViewChange<T>::expand({from, dataPtr()}, to));
  }

  BaseDataSP expand_(const Shape &from, const Shape &to) const final {
    return std::make_shared<ViewData<T>>(this->shared_from_this(),
                                         from.getExpandedRowMajorIndices(to));
  }

  BaseDataSP
  slice(const Shape &from, const Lower &l, const Upper &u) const final {
    return std::make_shared<AllocData<T>>(
        ViewChange<T>::slice({from, dataPtr()}, l, u));
  }

  BaseDataSP slice(const Shape &from,
                   const NormalizedSliceParams &n) const final {
    return std::make_shared<AllocData<T>>(
        ViewChange<T>::slice({from, dataPtr()}, n));
  }

  BaseDataSP gather(const Shape &from,
                    uint64_t dimension,
                    const std::vector<int64_t> &where) const final {
    return std::make_shared<AllocData<T>>(
        ViewChange<T>::gather({from, dataPtr()}, dimension, where));
  }

  BaseDataSP
  slice_(const Shape &from, const Lower &l, const Upper &u) const final {
    return std::make_shared<ViewData<T>>(this->shared_from_this(),
                                         from.getSlicedRowMajorIndices(l, u));
  }

  BaseDataSP slice_(const Shape &from,
                    const NormalizedSliceParams &n) const final {
    return std::make_shared<ViewData<T>>(this->shared_from_this(),
                                         from.getSlicedRowMajorIndices(n));
  }

  BaseDataSP gather_(const Shape &from,
                     uint64_t dimension,
                     const std::vector<int64_t> &indices) const final {
    return std::make_shared<ViewData<T>>(
        this->shared_from_this(),
        from.gatherRowMajorIndices(dimension, indices));
  }

  BaseDataSP dimShuffle(const Shape &from, const Permutation &p) const final {
    return std::make_shared<AllocData<T>>(
        ViewChange<T>::dimShuffle({from, dataPtr()}, p));
  }

  BaseDataSP dimShuffle_(const Shape &from,
                         const Permutation &p) const final {
    return std::make_shared<ViewData<T>>(
        this->shared_from_this(), from.getDimShuffledRowMajorIndices(p));
  }

  BaseDataSP reverse(const Shape &from,
                     const std::vector<uint64_t> &dims) const final {
    return std::make_shared<AllocData<T>>(
        ViewChange<T>::reverse({from, dataPtr()}, dims));
  }

  BaseDataSP reverse_(const Shape &from,
                      const std::vector<uint64_t> &dims) const final {
    return std::make_shared<ViewData<T>>(
        this->shared_from_this(), from.getReversedRowMajorIndices(dims));
  }

  BaseDataSP subSample(const Shape &from,
                       const std::vector<uint64_t> &strides) const final {
    return std::make_shared<AllocData<T>>(
        ViewChange<T>::subSample({from, dataPtr()}, strides));
  }

  BaseDataSP subSample_(const Shape &from,
                        const std::vector<uint64_t> &strides) const final {
    return std::make_shared<ViewData<T>>(
        this->shared_from_this(), from.getSubSampledRowMajorIndices(strides));
  }

  BaseDataSP toViewData_() const final {
    return std::make_shared<ViewData<T>>(
        this->shared_from_this(), OriginDataHelper::getIota_i64(nelms_u64()));
  }

  std::vector<double> getFloat64Vector() const final {
    return castToVector<double>();
  }

  std::vector<float> getFloat32Vector() const final {
    return castToVector<float>();
  }

  std::vector<uint16_t> getFloat16Vector_u16() const final {
    return OriginDataHelper::float16ToUint16(castToVector<IeeeHalf>());
  }

  std::vector<int64_t> getInt64Vector() const final {
    return castToVector<int64_t>();
  }

  std::vector<uint64_t> getUnsigned64Vector() const final {
    return castToVector<uint64_t>();
  }

  std::vector<int32_t> getInt32Vector() const final {
    return castToVector<int32_t>();
  }

  std::vector<uint32_t> getUnsigned32Vector() const final {
    return castToVector<uint32_t>();
  }

  std::vector<int16_t> getInt16Vector() const final {
    return castToVector<int16_t>();
  }

  std::vector<uint16_t> getUnsigned16Vector() const final {
    return castToVector<uint16_t>();
  }

  std::vector<int8_t> getInt8Vector() const final {
    return castToVector<int8_t>();
  }

  std::vector<uint8_t> getUnsigned8Vector() const final {
    return castToVector<uint8_t>();
  }

  std::vector<bool> getBoolVector() const final {
    return castToVector<bool>();
  }

  std::vector<T> getNativeVector() const final { return castToVector<T>(); }

  BaseDataSP toOriginData() const final {
    return std::make_shared<AllocData<T>>(std::move(castToVector<T>()));
  }

  BaseDataSP abs() const final { return unary<Abs<T>>(); }
  void abs_() const final { unary_<Abs<T>>(); }

  BaseDataSP sqrt() const final { return unary<Sqrt<T>>(); }
  void sqrt_() const final { unary_<Sqrt<T>>(); }

  BaseDataSP ceil() const final { return unary<Ceil<T>>(); }
  void ceil_() const final { unary_<Ceil<T>>(); }

  BaseDataSP floor() const final { return unary<Floor<T>>(); }
  void floor_() const final { unary_<Floor<T>>(); }

  BaseDataSP add(const BaseData &rhs) const final {
    auto out = binary<Adder<T>>(rhs);
    return out;
  }
  BaseDataSP mul(const BaseData &rhs) const final {
    return binary<Multiplier<T>>(rhs);
  }
  BaseDataSP pow(const BaseData &rhs) const final {
    return binary<Exponentiater<T>>(rhs);
  }
  BaseDataSP divide(const BaseData &rhs) const final {
    return binary<Divider<T>>(rhs);
  }
  BaseDataSP mod(const BaseData &rhs) const final {
    return binary<Modder<T>>(rhs);
  }
  BaseDataSP subtract(const BaseData &rhs) const final {
    return binary<Subtracter<T>>(rhs);
  }

  AllocBooleanSP greaterThan(const BaseData &rhs) const final {
    return binary<GreaterThan<T>, bool>(rhs);
  }
  AllocBooleanSP greaterThanOrEqualTo(const BaseData &rhs) const final {
    return binary<GreaterThanOrEqualTo<T>, bool>(rhs);
  }
  AllocBooleanSP lessThan(const BaseData &rhs) const final {
    return binary<LessThan<T>, bool>(rhs);
  }
  AllocBooleanSP lessThanOrEqualTo(const BaseData &rhs) const final {
    return binary<LessThanOrEqualTo<T>, bool>(rhs);
  }
  AllocBooleanSP equalTo(const BaseData &rhs) const final {
    return binary<EqualTo<T>, bool>(rhs);
  }

  void add_(const BaseData &rhs) const final { binary_<Adder<T>>(rhs); }
  void subtract_(const BaseData &rhs) const final {
    binary_<Subtracter<T>>(rhs);
  }
  void divide_(const BaseData &rhs) const final { binary_<Divider<T>>(rhs); }
  void mod_(const BaseData &rhs) const final { binary_<Modder<T>>(rhs); }
  void mul_(const BaseData &rhs) const final { binary_<Multiplier<T>>(rhs); }
  void pow_(const BaseData &rhs) const final {
    binary_<Exponentiater<T>>(rhs);
  }

  bool allZero() const final {
    const auto x0 = dataPtr();
    for (uint64_t i = 0; i < nelms_u64(); ++i) {
      if (x0[i] > T(0) || x0[i] < T(0)) {
        return false;
      }
    }
    return true;
  }

  bool allNonZero() const final {
    const auto x0 = dataPtr();
    for (uint64_t i = 0; i < nelms_u64(); ++i) {
      if (x0[i] >= T(0) && x0[i] <= T(0)) {
        return false;
      }
    }
    return true;
  }

private:
  template <class UnaryOp, class... Args>
  BaseDataSP unary(Args... args) const {
    const UnaryOp op(args...);
    Vec out(nelms_u64());

    const auto *srcBegin = dataPtr();
    const auto *srcEnd   = std::next(srcBegin, nelms_i64());
    auto dstBegin        = out.begin();
    std::transform(
        srcBegin, srcEnd, dstBegin, [op](auto x) { return op(x); });
    return std::make_shared<AllocData<T>>(std::move(out));
  }

  template <class UnaryOp, class... Args> void unary_(Args... args) const {
    const UnaryOp op(args...);
    std::for_each(
        dataPtr(), dataPtr() + nelms_u64(), [op](auto &x) { x = op(x); });
  }

  template <class BinaryOp> void binary_(const BaseData &rhs) const {
    const BinaryOp op;
    // const T *data_ = dataPtr();
    OriginDataHelper::assertSameBinaryOpNelms(
        rhs.nelms_u64(), nelms_u64(), *this);

    if (auto rhs_ = dynamic_cast<const OriginData<T> *>(&rhs)) {
      const auto *rhsData_ = rhs_->dataPtr();
      const auto *cDataPtr = dataPtr();
      std::transform(cDataPtr,
                     std::next(cDataPtr, nelms_i64()),
                     rhsData_,
                     dataPtr(),
                     [op](T a, T b) { return op(a, b); });
    } else {
      std::ostringstream oss;
      oss << "Call to " << *this << ".binary<" << BinaryOp::name() << ">("
          << rhs << ") failed.";
      throw error(oss.str());
    }
  }

  template <class BinaryOp, typename ReturnType = T>
  std::shared_ptr<AllocData<ReturnType>> binary(const BaseData &rhs) const {
    const BinaryOp op;
    OriginDataHelper::assertSameBinaryOpNelms(
        rhs.nelms_u64(), nelms_u64(), *this);
    if (auto rhs_ = dynamic_cast<const OriginData<T> *>(&rhs)) {
      const auto rhsData_  = rhs_->dataPtr();
      const auto thisData_ = dataPtr();
      std::vector<ReturnType> out;
      out.reserve(nelms_u64());
      for (uint64_t i = 0; i < nelms_u64(); ++i) {
        out.push_back(op(thisData_[i], rhsData_[i]));
      }

      return std::make_shared<AllocData<ReturnType>>(std::move(out));
    } else {
      std::ostringstream oss;
      oss << "Call to " << *this << ".binary<" << BinaryOp::name() << ">("
          << rhs << ") failed.";
      throw error(oss.str());
    }
  }

  template <typename To> std::vector<To> castToVector() const {
    return castPtrToVector<T, To>(dataPtr(), nelms_u64());
  }

  template <typename U> BaseDataSP castToBaseDataSP() const {
    return std::make_shared<AllocData<U>>(
        castPtrToVector<T, U>(dataPtr(), nelms_u64()));
  }
};

} // namespace host
} // namespace compute
} // namespace poprithms

#endif
