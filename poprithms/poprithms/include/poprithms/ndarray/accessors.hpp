// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_NDARRAY_ACCESSORS_HPP
#define POPRITHMS_NDARRAY_ACCESSORS_HPP

#include <array>
#include <ostream>
#include <vector>

/**
 * These structs wrap integer and std::vector types, and are used to
 * safeguard against user bugs arising from accidentally permuting arguments
 * to methods with multiple inputs of the same type.
 * */

namespace poprithms {
namespace ndarray {

struct BaseScalarU64 {
  BaseScalarU64(uint64_t v_) : val(v_) {}
  uint64_t get() const { return val; }
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

struct Dilations : public BaseVectorU64 {
  explicit Dilations(const std::vector<uint64_t> &d);
  explicit Dilations(std::vector<uint64_t> &&d);
  Dilations() : BaseVectorU64() {}
  explicit Dilations(const std::vector<Dilation> &);
  Dilation at(uint64_t d) const { return Dilation(vals[d]); }
};

struct Strides : public BaseVectorU64 {
  explicit Strides(const std::vector<uint64_t> &d);
  explicit Strides(std::vector<uint64_t> &&d);
  Strides() : BaseVectorU64() {}
  explicit Strides(const std::vector<Stride> &);
  Stride at(uint64_t d) const { return Stride(vals[d]); }
};

} // namespace ndarray
} // namespace poprithms

#endif
