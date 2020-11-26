// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMPUTE_HOST_VIEWDATA_HPP
#define POPRITHMS_COMPUTE_HOST_VIEWDATA_HPP
#include "baseoperators.hpp"
#include "gridpointhelper.hpp"
#include "typeddata.hpp"

#include <poprithms/compute/host/viewchange.hpp>

namespace poprithms {
namespace compute {
namespace host {

/**
 * A reference to a view of OriginData BaseDatas.
 *
 * An element in a ViewData BaseData is represented by
 *   1) a pointer to a OriginData BaseData, or an "origin", and
 *   2) an offset from the zero'th element of the OriginData BaseData.
 *
 * While a ViewData behaves like a C++-style reference, the implementation is
 * not memory efficient, as every underlying element is recorded by 2 values
 * ((1) and (2) above).
 *
 * */
template <class T>
class ViewData : public TypedData<T>,
                 public std::enable_shared_from_this<ViewData<T>> {

private:
  using OriginDatas = std::vector<std::shared_ptr<const OriginData<T>>>;

  // The BaseDatas where the underlying data lives. It is not required that
  // all of the OriginDatas in this vector are referenced by this
  // ViewData. Note that by storing the shared_ptr's, and not just the
  // raw pointers, it is guaranteed that the OriginData Tensors are live as
  // long as this ViewData is live.
  OriginDatas rowMajorOriginDatas;
  uint64_t nOriginDatas() const { return rowMajorOriginDatas.size(); }

  // The pointers to the first elements of the BaseDatas where the
  // underlying data lives. The raw pointers are stored in addition to the
  // shared_ptrs in rowMajorOriginDatas, to avoid multiple virtual method
  // calls into polymorphic OriginData objects.
  std::vector<T *> rowMajorOriginDataPtrs;

  // A vector of length nelms(), which stores the underlying BaseData where
  // particular Tensor elements live. For example, if rowMajorOriginDatas[5] =
  // 3, then the 5'th element of this BaseData is in rowMajorOriginDatas[3]
  std::vector<uint64_t> rowMajorOriginDataIndices;
  T *dataPtr0(uint64_t i) const {
    return rowMajorOriginDataPtrs[rowMajorOriginDataIndices[i]];
  }

  // A vector of length nelms(), which stores the index within the underlying
  // data array for particular elements. For example, if
  //   rowMajorOriginDataIndices[5] = 3, and
  //   rowMajorOriginDataOffsets[5] = 6, then the 5'th element of this
  //   BaseData is rowMajorOriginDataPtrs[3][6].
  std::vector<int64_t> rowMajorOriginDataOffsets;

  // The address of underlying data of the i'th element of this BaseData
  T *dataPtr(uint64_t i) const {
    auto ptr = std::next(dataPtr0(i), rowMajorOriginDataOffsets[i]);
    return ptr;
  }

  bool allZero() const final {
    for (uint64_t i = 0; i < nelms_u64(); ++i) {
      if (*dataPtr(i) > T(0) || *dataPtr(i) < T(0)) {
        return false;
      }
    }
    return true;
  }

  bool allNonZero() const final {
    for (uint64_t i = 0; i < nelms_u64(); ++i) {
      if (*dataPtr(i) >= T(0) && *dataPtr(i) <= T(0)) {
        return false;
      }
    }
    return true;
  }

public:
  using BaseData::nelms_u64;

  const auto &indices() const { return rowMajorOriginDataIndices; }

  const auto &offsets() const { return rowMajorOriginDataOffsets; }

  const OriginDatas &origins() const { return rowMajorOriginDatas; }

  /**
   * Reset \p rowMajorOriginDatas to be \p nxt, and update \p
   * rowMajorOriginDataPtrs and \p rowMajorOriginDataIndices accordingly.
   *
   * \p nxt must be a supserset \p rowMajorOriginDatas, otherwise an error is
   * thrown.
   *
   * This transformation only alters the internal representation of this
   * ViewData -- any operations on this object are unaffected by this call. It
   * is used to canonicalize ViewDatas when they are concatenated.
   *
   * \see TypedConcat_
   * */
  void remapOriginDatas(const OriginDatas &nxt) {

    (void)nxt;

    // a map from current origins to the indices where they appear in
    // nxt. For example, if toNxt[2] = 5, then
    // rowMajorOriginDatas[2] = nxt[5].
    std::vector<uint64_t> toNxt(nOriginDatas());
    for (uint64_t i = 0; i < nOriginDatas(); ++i) {
      const auto found =
          std::find(nxt.cbegin(), nxt.cend(), rowMajorOriginDatas[i]);
      if (found == nxt.cend()) {
        std::ostringstream oss;
        oss << "Error in " << *this << ".remapOriginDatas(.): "
            << "not all OriginDatas in rowMajorOriginDatas appear in nxt. ";
        throw error(oss.str());
      }
      toNxt[i] = static_cast<uint64_t>(std::distance(nxt.cbegin(), found));
    }

    std::transform(rowMajorOriginDataIndices.cbegin(),
                   rowMajorOriginDataIndices.cend(),
                   rowMajorOriginDataIndices.begin(),
                   [&toNxt](auto x) { return toNxt[x]; });

    rowMajorOriginDatas = nxt;
    rowMajorOriginDataPtrs.clear();
    for (auto o : rowMajorOriginDatas) {
      rowMajorOriginDataPtrs.push_back(o->dataPtr());
    }
  }

  ViewData(const OriginDatas &origins_,
           std::vector<uint64_t> &&indices_,
           std::vector<int64_t> &&offsets_)
      : rowMajorOriginDatas(origins_),
        rowMajorOriginDataIndices(std::move(indices_)),
        rowMajorOriginDataOffsets(std::move(offsets_)) {
    rowMajorOriginDataPtrs.reserve(rowMajorOriginDatas.size());
    for (auto o : rowMajorOriginDatas) {
      rowMajorOriginDataPtrs.push_back(o->dataPtr());
    }
  }

  ViewData(const ViewData &) = default;
  ViewData(ViewData &&)      = default;
  ViewData &operator=(const ViewData &) = default;
  ViewData &operator=(ViewData &&) = default;

  ViewData(std::shared_ptr<const OriginData<T>> rm,
           std::vector<int64_t> &&offsets_)
      : ViewData(OriginDatas{rm},
                 std::vector<uint64_t>(offsets_.size(), 0),
                 std::move(offsets_)) {}

  bool isOriginData() const final { return false; }

  std::shared_ptr<BaseData> expand(const Shape &from,
                                   const Shape &to) const final {
    return toOriginData()->expand(from, to);
  }

  BaseDataSP
  slice(const Shape &from, const Lower &l, const Upper &u) const final {
    return toOriginData()->slice(from, l, u);
  }

  BaseDataSP gather(const Shape &from,
                    uint64_t dimension,
                    const std::vector<int64_t> &where) const final {
    return toOriginData()->gather(from, dimension, where);
  }

  BaseDataSP
  slice_(const Shape &from, const Lower &l, const Upper &u) const final {
    return std::make_shared<ViewData<T>>(
        origins(),
        ViewChange<uint64_t>::slice(
            {from, rowMajorOriginDataIndices.data()}, l, u),
        ViewChange<int64_t>::slice(
            {from, rowMajorOriginDataOffsets.data()}, l, u));
  }

  BaseDataSP gather_(const Shape &from,
                     uint64_t dimension,
                     const std::vector<int64_t> &where) const final {
    return std::make_shared<ViewData<T>>(
        origins(),
        ViewChange<uint64_t>::gather(
            {from, rowMajorOriginDataIndices.data()}, dimension, where),
        ViewChange<int64_t>::gather(
            {from, rowMajorOriginDataOffsets.data()}, dimension, where));
  }

  BaseDataSP expand_(const Shape &from, const Shape &to) const final {
    return std::make_shared<ViewData<T>>(
        origins(),
        ViewChange<uint64_t>::expand({from, rowMajorOriginDataIndices.data()},
                                     to),
        ViewChange<int64_t>::expand({from, rowMajorOriginDataOffsets.data()},
                                    to));
  }

  BaseDataSP dimShuffle(const Shape &from, const Permutation &p) const final {
    return toOriginData()->dimShuffle(from, p);
  }

  BaseDataSP dimShuffle_(const Shape &from,
                         const Permutation &p) const final {
    return std::make_shared<ViewData<T>>(
        origins(),
        ViewChange<uint64_t>::dimShuffle(
            {from, rowMajorOriginDataIndices.data()}, p),
        ViewChange<int64_t>::dimShuffle(
            {from, rowMajorOriginDataOffsets.data()}, p));
  }

  BaseDataSP reverse(const Shape &from,
                     const std::vector<uint64_t> &dims) const final {
    return toOriginData()->reverse(from, dims);
  }

  BaseDataSP reverse_(const Shape &from,
                      const std::vector<uint64_t> &dims) const final {
    return std::make_shared<ViewData<T>>(
        origins(),
        ViewChange<uint64_t>::reverse(
            {from, rowMajorOriginDataIndices.data()}, dims),
        ViewChange<int64_t>::reverse({from, rowMajorOriginDataOffsets.data()},
                                     dims));
  }

  BaseDataSP subSample(const Shape &from,
                       const std::vector<uint64_t> &strides) const final {
    return toOriginData()->subSample(from, strides);
  }

  BaseDataSP subSample_(const Shape &from,
                        const std::vector<uint64_t> &strides) const final {
    return std::make_shared<ViewData<T>>(
        origins(),
        ViewChange<uint64_t>::subSample(
            {from, rowMajorOriginDataIndices.data()}, strides),
        ViewChange<int64_t>::subSample(
            {from, rowMajorOriginDataOffsets.data()}, strides));
  }

  std::vector<uint16_t> getFloat16Vector_u16() const final {
    return toOriginData()->getFloat16Vector_u16();
  }

  std::vector<char> getNativeCharVector() const final {
    return toOriginData()->getNativeCharVector();
  }

  std::vector<double> getFloat64Vector() const final {
    return getVector<double>();
  }

  std::vector<float> getFloat32Vector() const final {
    return getVector<float>();
  }

  std::vector<int32_t> getInt32Vector() const final {
    return getVector<int32_t>();
  }
  std::vector<uint32_t> getUnsigned32Vector() const final {
    return getVector<uint32_t>();
  }

  std::vector<int64_t> getInt64Vector() const final {
    return getVector<int64_t>();
  }
  std::vector<uint64_t> getUnsigned64Vector() const final {
    return getVector<uint64_t>();
  }

  std::vector<int16_t> getInt16Vector() const final {
    return getVector<int16_t>();
  }

  std::vector<uint16_t> getUnsigned16Vector() const final {
    return getVector<uint16_t>();
  }

  std::vector<int8_t> getInt8Vector() const final {
    return getVector<int8_t>();
  }

  std::vector<uint8_t> getUnsigned8Vector() const final {
    return getVector<uint8_t>();
  }

  std::vector<bool> getBoolVector() const final {
    std::vector<bool> out;
    out.reserve(nelms_u64());
    for (auto x : getNativeVector()) {
      out.push_back(x);
    }
    return out;
  }

  BaseDataSP toOriginData() const final { return cast<T>(); }

  BaseDataSP abs() const final { return unary<Abs<T>>(); }
  void abs_() const final { unary_<Abs<T>>(); }

  BaseDataSP sqrt() const final { return unary<Sqrt<T>>(); }
  void sqrt_() const final { unary_<Sqrt<T>>(); }

  BaseDataSP ceil() const final { return unary<Ceil<T>>(); }
  void ceil_() const final { unary_<Ceil<T>>(); }

  BaseDataSP floor() const final { return unary<Floor<T>>(); }
  void floor_() const final { unary_<Floor<T>>(); }

  BaseDataSP add(const BaseData &rhs) const final {
    return toOriginData()->add(rhs);
  }
  BaseDataSP mul(const BaseData &rhs) const final {
    return toOriginData()->mul(rhs);
  }
  BaseDataSP pow(const BaseData &rhs) const final {
    return toOriginData()->pow(rhs);
  }
  BaseDataSP divide(const BaseData &rhs) const final {
    return toOriginData()->divide(rhs);
  }
  BaseDataSP mod(const BaseData &rhs) const final {
    return toOriginData()->mod(rhs);
  }
  BaseDataSP subtract(const BaseData &rhs) const final {
    return toOriginData()->subtract(rhs);
  }

  AllocBooleanSP greaterThan(const BaseData &rhs) const final {
    return toOriginData()->greaterThan(rhs);
  }
  AllocBooleanSP greaterThanOrEqualTo(const BaseData &rhs) const final {
    return toOriginData()->greaterThanOrEqualTo(rhs);
  }
  AllocBooleanSP lessThan(const BaseData &rhs) const final {
    return toOriginData()->lessThan(rhs);
  }
  AllocBooleanSP lessThanOrEqualTo(const BaseData &rhs) const final {
    return toOriginData()->lessThanOrEqualTo(rhs);
  }
  AllocBooleanSP equalTo(const BaseData &rhs) const final {
    return toOriginData()->equalTo(rhs);
  }

  uint64_t nelms_u64() const final {
    return rowMajorOriginDataIndices.size();
  }

  BaseDataSP clone() const final {
    return std::make_shared<ViewData<T>>(*this);
  }

  void append(std::ostream &ost) const final {
    ost << "ViewData(dtype=" << poprithms::ndarray::lcase<T>()
        << ",nelms=" << nelms_u64() << ')';
  }

  BaseDataSP toViewData_() const final { return clone(); }

  void add_(const BaseData &rhs) const final { binary_<Adder<T>>(rhs); }

  void subtract_(const BaseData &rhs) const final {
    binary_<Subtracter<T>>(rhs);
  }

  void divide_(const BaseData &rhs) const final { binary_<Divider<T>>(rhs); }

  void mod_(const BaseData &rhs) const final { binary_<Modder<T>>(rhs); }

  void mul_(const BaseData &rhs) const final { binary_<Multiplier<T>>(rhs); }

  void pow_(const BaseData &rhs) const final { binary_<Exponentiater<T>>(rhs); }

  bool containsAliases() const final {
    return !GridPointHelper::allUnique(indices(), offsets());
  }

private:
  template <class UnaryOp, class... Args> void unary_(Args... args) const {

    // get all unique elements (remove duplicates) in origins, and apply the
    // UnaryOp to them. Note that this behaviour is unlike poplar, where an
    // error is thrown if a Tensor contains aliases.
    const UnaryOp op(args...);
    const auto singles = GridPointHelper::getUnique(indices(), offsets());

    std::for_each(singles.cbegin(), singles.cend(), [this, op](auto single) {
      auto index  = std::get<0>(single);
      auto offset = std::get<1>(single);
      auto &v     = rowMajorOriginDataPtrs[index][offset];
      v           = op(v);
    });
  }

  template <class BinaryOp> void binary_(const BaseData &rhs) const {
    const BinaryOp op;
    if (containsAliases()) {
      throw error("ViewData::binary_ not implemented for self-aliases");
    }
    if (auto rhs_ = dynamic_cast<const OriginData<T> *>(&rhs)) {
      const auto *rhsData_ = rhs_->dataPtr();
      for (uint64_t i = 0; i < nelms_u64(); ++i) {
        *dataPtr(i) = op(*dataPtr(i), rhsData_[i]);
      }
    } else {
      std::ostringstream oss;
      oss << "Call to " << *this << ".binary_<" << BinaryOp::name() << ">("
          << rhs << ") failed. "
          << "Note that binary_ does not currently support "
          << "a rhs which is a ViewData. ";
      throw error(oss.str());
    }
  }

  template <class UnaryOp, class... Args>
  std::vector<T> unaryVector(Args... args) const {
    const UnaryOp op(args...);
    std::vector<T> out(nelms_u64());
    for (uint64_t i = 0; i < out.size(); ++i) {
      out[i] = op(*dataPtr(i));
    }
    return out;
  }

  template <class UnaryOp, class... Args>
  BaseDataSP unary(Args... args) const {
    return std::make_shared<AllocData<T>>(unaryVector<UnaryOp>(args...));
  }

  template <typename To> std::vector<To> getVector() const {
    auto v0 = getNativeVector();
    std::vector<To> out(nelms_u64());
    std::transform(v0.cbegin(), v0.cend(), out.begin(), [](auto x) {
      return static_cast<To>(x);
    });
    return out;
  }

  template <typename To> std::shared_ptr<OriginData<To>> cast() const {
    return std::make_shared<AllocData<To>>(getVector<To>());
  }

public:
  std::vector<T> getNativeVector() const final {
    std::vector<T> out(nelms_u64());
    for (uint64_t i = 0; i < out.size(); ++i) {
      out[i] = *dataPtr(i);
    }
    return out;
  }
};

} // namespace host
} // namespace compute
} // namespace poprithms

#endif
