// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef POPRITHMS_COMMON_COMPUTE_DEVICE_HPP
#define POPRITHMS_COMMON_COMPUTE_DEVICE_HPP

#include <poprithms/common/compute/devicetype.hpp>
#include <poprithms/ndarray/deviceid.hpp>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/ndarray/shape.hpp>

namespace poprithms {
namespace common {
namespace compute {

using poprithms::ndarray::DeviceId;
using poprithms::ndarray::DeviceIds;
using poprithms::ndarray::DType;
using poprithms::ndarray::DTypes;
using poprithms::ndarray::Shape;
using poprithms::ndarray::Shapes;

/**
 * Representation of a device where a tensor is located.
 * */
class Device {

public:
  Device() = delete;

  /**
   * Construct a device with id #id of device type #deviceType.
   * */
  Device(DeviceId id, DeviceType deviceType)
      : id_(id), deviceType_(deviceType) {}

  virtual ~Device() = default;

  DeviceId id() const { return id_; }
  DeviceType deviceType() const { return deviceType_; }

  /**
   * Can this Device store a tensor of shape #s and numerical type #type?
   * */
  bool canStore(const Shape &s, DType type) const {
    return canStoreShape(s) && canStoreDType(type);
  }

  /**
   * Throw a descriptive error if this device cannot store a tensor of shape
   * #s and numerical type #dt.
   * */
  void confirmCanStore(const Shape &s, DType dt) const;

  bool isIpu() const { return deviceType_ == DeviceType::Ipu; }
  bool isHost() const { return deviceType_ == DeviceType::Host; }
  bool isRemote() const { return deviceType_ == DeviceType::Remote; }

  /**
   * We assume that every device created for an application can be identified
   * by a unique id. The application must manage this.
   * */
  bool operator==(const Device &d) const { return id_ == d.id_; }
  bool operator!=(const Device &d) const { return !operator==(d); }

  std::string str() const;

  virtual std::unique_ptr<Device> clone() const = 0;

  /**
   * Return true if this device can store a tensor of shape #s.
   * */
  virtual bool canStoreShape(const Shape &s) const = 0;

  /**
   * Return true if this device can store a tensor of numerical type #dt.
   * */
  virtual bool canStoreDType(DType dt) const = 0;

private:
  DeviceId id_;
  /*
   * The type of the device. Design note: it's not great design having this
   * enum attribute stored in a base class, but currently it's very useful and
   * unlike ops where we expect multiple new ones to be added with different
   * semantics, we don't expect many more device types to be added.
   * */
  DeviceType deviceType_;
};

/**
 * The host device type.
 * */
class Host : public Device {
public:
  Host(DeviceId id) : Device(id, DeviceType::Host) {}
  std::unique_ptr<Device> clone() const final;

private:
  /**
   * There is no restriction on the shape or type of host tensor, so these
   * methods return true.
   * */
  bool canStoreShape(const Shape &) const final { return true; }
  bool canStoreDType(DType) const final { return true; }
};

std::ostream &operator<<(std::ostream &oss, const Device &);

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
