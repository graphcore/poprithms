// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_NDARRAY_ACCESSORS_HPP
#define POPRITHMS_NDARRAY_ACCESSORS_HPP

#include <algorithm>
#include <array>
#include <initializer_list>
#include <ostream>
#include <vector>

/**
 * Structs which wrap integer and std::vector types, and can be used to
 * safeguard against user bugs arising from accidentally permuting arguments
 * to methods with multiple inputs of the same type.
 * */
namespace poprithms {
namespace ndarray {

template <typename T>
std::vector<uint64_t> get_u64(const std::vector<T> &ds) {
  std::vector<uint64_t> vs;
  vs.reserve(ds.size());
  for (auto x : ds) {
    vs.push_back(x.get());
  }
  return vs;
}

struct BaseScalarU64 {
  BaseScalarU64(uint64_t v_) : val(v_) {}
  uint64_t get() const { return val; }
  int64_t get_i64() const { return static_cast<int64_t>(val); }
  uint64_t val;
};

/** Strictly positive value */
struct Stride : BaseScalarU64 {
  explicit Stride(uint64_t s_);
};

struct Dimension : BaseScalarU64 {
  explicit Dimension(uint64_t s_) : BaseScalarU64(s_) {}
};

struct Dilation : BaseScalarU64 {
  explicit Dilation(uint64_t s_);
};

/**
 * Using the curiously recurring template pattern (CRTP) to reduce code
 * duplication. The template parameter D is the leaf class which derives from
 * BaseVector.
 * */
template <typename T, typename D> struct BaseVector {

  BaseVector() = default;
  BaseVector(const std::vector<T> &vs_) : vals(vs_) {}
  BaseVector(std::vector<T> &&vs_) : vals(std::move(vs_)) {}
  std::vector<T> get() const { return vals; }
  uint64_t size() const { return vals.size(); }
  bool empty() const { return vals.empty(); }
  bool operator==(const BaseVector<T, D> &rhs) const {
    return vals == rhs.vals;
  }
  bool operator!=(const BaseVector<T, D> &rhs) const {
    return !operator==(rhs);
  }

  D sorted() const {
    auto a = get();
    std::sort(a.begin(), a.end());
    return D(a);
  }

  /** Concatenate the Dimensions in \a rhs to these Dimensions. */
  D append(const BaseVector<T, D> &rhs) const {
    auto a       = get();
    const auto b = rhs.get();
    a.insert(a.end(), b.cbegin(), b.cend());
    return D(a);
  }

  std::vector<T> vals;
};

struct Starts : public BaseVector<int64_t, Starts> {
  explicit Starts(const std::vector<int64_t> &s)
      : BaseVector<int64_t, Starts>(s) {}
  Starts() : BaseVector<int64_t, Starts>() {}
  explicit Starts(const std::vector<int64_t> &&s)
      : BaseVector<int64_t, Starts>(std::move(s)) {}
};

struct Ends : public BaseVector<int64_t, Ends> {
  explicit Ends(const std::vector<int64_t> &s)
      : BaseVector<int64_t, Ends>(s) {}
  Ends() : BaseVector<int64_t, Ends>() {}
  explicit Ends(const std::vector<int64_t> &&s)
      : BaseVector<int64_t, Ends>(std::move(s)) {}
};

struct Dims : public BaseVector<int64_t, Dims> {
  explicit Dims(const std::vector<int64_t> &s)
      : BaseVector<int64_t, Dims>(s) {}
  Dims() : BaseVector<int64_t, Dims>() {}
  explicit Dims(const std::vector<int64_t> &&s)
      : BaseVector<int64_t, Dims>(std::move(s)) {}
};

struct Steps : public BaseVector<int64_t, Steps> {
  explicit Steps(const std::vector<int64_t> &s)
      : BaseVector<int64_t, Steps>(s) {}
  Steps() : BaseVector<int64_t, Steps>() {}
  explicit Steps(const std::vector<int64_t> &&s)
      : BaseVector<int64_t, Steps>(std::move(s)) {}
};

template <typename T, typename D, typename S>
struct VU64 : public BaseVector<T, D> {
  VU64() : BaseVector<T, D>() {}
  explicit VU64(const std::vector<T> &d) : BaseVector<T, D>(d) {}
  explicit VU64(std::initializer_list<T> d) : BaseVector<T, D>(d) {}
  explicit VU64(std::vector<T> &&d) : BaseVector<T, D>(std::move(d)) {}
  explicit VU64(const std::vector<D> &d) : VU64(get_u64(d)) {}
  S at(uint64_t d) const { return S(BaseVector<T, D>::vals[d]); }
};

struct Strides : public VU64<uint64_t, Strides, Stride> {
  using Base = VU64<uint64_t, Strides, Stride>;
  Strides() : Base() {}
  explicit Strides(const std::vector<uint64_t> &d) : Base(d) {}
  explicit Strides(std::initializer_list<uint64_t> d) : Base(d) {}
  explicit Strides(std::vector<uint64_t> &&d) : Base(std::move(d)) {}
  explicit Strides(const std::vector<Stride> &d) : Strides(get_u64(d)) {}
};
std::ostream &operator<<(std::ostream &, const Strides &);

struct Dilations : public VU64<uint64_t, Dilations, Dilation> {
  using Base = VU64<uint64_t, Dilations, Dilation>;
  Dilations() : Base() {}
  explicit Dilations(const std::vector<uint64_t> &d) : Base(d) {}
  explicit Dilations(std::initializer_list<uint64_t> d) : Base(d) {}
  explicit Dilations(std::vector<uint64_t> &&d) : Base(std::move(d)) {}
  explicit Dilations(const std::vector<Dilation> &d)
      : Dilations(get_u64(d)) {}
};
std::ostream &operator<<(std::ostream &, const Dilations &);

struct Dimensions : public VU64<uint64_t, Dimensions, Dimension> {
  using Base = VU64<uint64_t, Dimensions, Dimension>;
  Dimensions() : Base() {}
  explicit Dimensions(const std::vector<uint64_t> &d) : Base(d) {}
  explicit Dimensions(std::initializer_list<uint64_t> d) : Base(d) {}
  explicit Dimensions(std::vector<uint64_t> &&d) : Base(std::move(d)) {}
  explicit Dimensions(const std::vector<Dimension> &d)
      : Dimensions(get_u64(d)) {}
};
std::ostream &operator<<(std::ostream &, const Dimensions &);
std::ostream &operator<<(std::ostream &, const std::vector<Dimensions> &);

} // namespace ndarray
} // namespace poprithms

#endif
