// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_TENSORINFO_HPP
#define POPRITHMS_COMMON_COMPUTE_TENSORINFO_HPP

#include <tuple>
#include <vector>

#include <poprithms/ndarray/deviceid.hpp>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/ndarray/shape.hpp>

namespace poprithms {
namespace ndarray {

using poprithms::ndarray::DType;
using poprithms::ndarray::DTypes;
using poprithms::ndarray::Shape;
using poprithms::ndarray::Shapes;

/**
 * A convenience class for handling the triplet (Shape, DeviceId, DType).
 * */
class TensorInfo {
public:
  TensorInfo(const Shape &s, DeviceId i, DType t)
      : shape_(s), deviceId_(i), dtype_(t) {}

  TensorInfo(Shape &&s, DeviceId i, DType t)
      : shape_(std::move(s)), deviceId_(i), dtype_(t) {}

  TensorInfo(const TensorInfo &) = default;
  TensorInfo(TensorInfo &&)      = default;

  TensorInfo &operator=(const TensorInfo &) = default;
  TensorInfo &operator=(TensorInfo &&)      = default;

  TensorInfo()  = delete;
  ~TensorInfo() = default;

  bool operator==(const TensorInfo &rhs) const { return tup() == rhs.tup(); }
  bool operator!=(const TensorInfo &rhs) const { return !operator==(rhs); }
  bool operator<(const TensorInfo &rhs) const { return tup() < rhs.tup(); }
  bool operator>(const TensorInfo &rhs) const { return tup() > rhs.tup(); }
  bool operator<=(const TensorInfo &rhs) const { return tup() <= rhs.tup(); }
  bool operator>=(const TensorInfo &rhs) const { return tup() >= rhs.tup(); }

  /**
   * Return a copy of this TensorInfo, but with Shape #s.
   * */
  TensorInfo withShape(const Shape &s) const &;
  TensorInfo withShape(const Shape &s) &&;

  /**
   * Return a copy of this TensorInfo, but with DeviceId #d.
   * */
  TensorInfo withDeviceId(DeviceId d) const &;
  TensorInfo withDeviceId(DeviceId d) &&;

  /**
   * Return a copy of this TensorInfo but with numerical type #dt.
   * */
  TensorInfo withDType(DType dt) const &;
  TensorInfo withDType(DType dt) &&;

  /**
   * Getters for Shape, DeviceId, and DType:
   * */
  const Shape &shape() const { return shape_; }
  DeviceId deviceId() const { return deviceId_; }
  DType dtype() const { return dtype_; }

  void append(std::ostream &) const;

private:
  Shape shape_;
  DeviceId deviceId_;
  DType dtype_;

  std::tuple<const Shape &, DeviceId, DType> tup() const {
    return {shape_, deviceId_, dtype_};
  }
};

std::ostream &operator<<(std::ostream &, const TensorInfo &);

/**
 * A sequence of objects of type TensorInfo, with utility methods to get
 * (unzip) vectors of their shapes, ids, and types.
 * */
class TensorInfos {

public:
  TensorInfos() = default;

  TensorInfos(const TensorInfo &t)
      : TensorInfos(std::vector<TensorInfo>({t})) {}

  TensorInfos(const std::vector<TensorInfo> &ts) : tensorInfos_(ts) {}

  TensorInfos(std::vector<TensorInfo> &&ts) : tensorInfos_(std::move(ts)) {}

  bool operator==(const TensorInfos &rhs) const {
    return tensorInfos_ == rhs.tensorInfos_;
  }

  bool operator!=(const TensorInfos &rhs) const { return !operator==(rhs); }

  /**
   * The Shapes of each of the TensorInfos.
   * */
  Shapes shapes() const;

  /**
   * The DeviceIds of each of the TensorInfos.
   * */
  DeviceIds deviceIds() const;

  /**
   * The DTypes of each of the TensorInfos.
   * */
  DTypes dtypes() const;

  size_t size() const { return tensorInfos_.size(); }

  void append(std::ostream &) const;

private:
  std::vector<TensorInfo> tensorInfos_;
};

std::ostream &operator<<(std::ostream &, const TensorInfos &);

} // namespace ndarray
} // namespace poprithms

#endif
