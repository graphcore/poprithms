// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_NDARRAY_ACCESSORS_HPP
#define POPRITHMS_NDARRAY_ACCESSORS_HPP

#include <array>
#include <initializer_list>
#include <ostream>
#include <vector>

/**
 * These structs wrap integer and std::vector types, and are used to
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

template <typename T> struct BaseVector {
  BaseVector(const std::vector<T> &vs_) : vals(vs_) {}
  BaseVector() = default;
  BaseVector(std::vector<T> &&vs_) : vals(std::move(vs_)) {}
  std::vector<T> get() const { return vals; }
  std::vector<T> vals;
  uint64_t size() const { return vals.size(); }
  bool empty() const { return vals.empty(); }
  bool operator==(const BaseVector<T> &rhs) const { return vals == rhs.vals; }
  bool operator!=(const BaseVector<T> &rhs) const { return !operator==(rhs); }
};

using BaseVectorI64 = BaseVector<int64_t>;
using BaseVectorU64 = BaseVector<uint64_t>;

struct Starts : public BaseVectorI64 {
  explicit Starts(const std::vector<int64_t> &s) : BaseVectorI64(s) {}
  Starts() : BaseVectorI64() {}
  explicit Starts(const std::vector<int64_t> &&s)
      : BaseVectorI64(std::move(s)) {}
};

struct Ends : public BaseVectorI64 {
  explicit Ends(const std::vector<int64_t> &s) : BaseVectorI64(s) {}
  Ends() : BaseVectorI64() {}
  explicit Ends(const std::vector<int64_t> &&s)
      : BaseVectorI64(std::move(s)) {}
};

struct Dims : public BaseVectorI64 {
  explicit Dims(const std::vector<int64_t> &s) : BaseVectorI64(s) {}
  Dims() : BaseVectorI64() {}
  explicit Dims(const std::vector<int64_t> &&s)
      : BaseVectorI64(std::move(s)) {}
};

struct Steps : public BaseVectorI64 {
  explicit Steps(const std::vector<int64_t> &s) : BaseVectorI64(s) {}
  Steps() : BaseVectorI64() {}
  explicit Steps(const std::vector<int64_t> &&s)
      : BaseVectorI64(std::move(s)) {}
};

struct Strides : public BaseVectorU64 {
  Strides() : BaseVectorU64() {}
  explicit Strides(const std::vector<uint64_t> &d) : BaseVectorU64(d) {}
  explicit Strides(std::initializer_list<uint64_t> d) : BaseVectorU64(d) {}
  explicit Strides(std::vector<uint64_t> &&d) : BaseVectorU64(std::move(d)) {}
  explicit Strides(const std::vector<Stride> &d) : Strides(get_u64(d)) {}
  Stride at(uint64_t d) const { return Stride(vals[d]); }
};
std::ostream &operator<<(std::ostream &, const Stride &);

struct Dilations : public BaseVectorU64 {
  Dilations() : BaseVectorU64() {}
  explicit Dilations(const std::vector<uint64_t> &d) : BaseVectorU64(d) {}
  explicit Dilations(std::initializer_list<uint64_t> d) : BaseVectorU64(d) {}
  explicit Dilations(std::vector<uint64_t> &&d)
      : BaseVectorU64(std::move(d)) {}
  explicit Dilations(const std::vector<Dilation> &d)
      : Dilations(get_u64(d)) {}
  Dilation at(uint64_t d) const { return Dilation(vals[d]); }
};
std::ostream &operator<<(std::ostream &, const Dilations &);

struct Dimensions : public BaseVectorU64 {
  Dimensions() : BaseVectorU64() {}
  explicit Dimensions(const std::vector<uint64_t> &d) : BaseVectorU64(d) {}
  explicit Dimensions(std::initializer_list<uint64_t> d) : BaseVectorU64(d) {}
  explicit Dimensions(std::vector<uint64_t> &&d)
      : BaseVectorU64(std::move(d)) {}
  explicit Dimensions(const std::vector<Dimension> &d)
      : Dimensions(get_u64(d)) {}
  Dimension at(uint64_t d) const { return Dimension(vals[d]); }

  /** Concatenate the Dimensions in \a rhs to these Dimensions. */
  Dimensions append(const Dimensions &rhs) const;
};
std::ostream &operator<<(std::ostream &, const Dimensions &);
std::ostream &operator<<(std::ostream &, const std::vector<Dimensions> &);

} // namespace ndarray
} // namespace poprithms

#endif
