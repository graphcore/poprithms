// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef POPRITHMS_COMMON_COMPUTE_REMOTE_HPP
#define POPRITHMS_COMMON_COMPUTE_REMOTE_HPP

#include <poprithms/common/compute/device.hpp>
#include <poprithms/common/compute/remoteoptions.hpp>

namespace poprithms {
namespace common {
namespace compute {

/**
 * A remote device, where tensors can be stored but not computed with. Each
 * remote device has a single associated ipu device. This device class
 * corresponds to a poplar remote buffer.
 * */
class Remote : public Device {
public:
  /**
   *
   * \param remoteId The id of the remote device being created.
   *
   * \param ipuId The id of the ipu which the remote device is associated to.
   *
   * \param type The numerical type of the tensor stored on the remote device.
   *
   * \param shape Rank-2 with elements (repeats, numElements). See
   *              poplar::Graph::addRemoteBuffer for details.
   * */
  Remote(DeviceId remoteId,
         DeviceId ipuId,
         DType type,
         const Shape &shape,
         const RemoteOptions &ros = {});

  std::unique_ptr<Device> clone() const final;

  /**
   * The number of elements transferred in a copy to/from a remote device.
   * See poplar::Graph::addRemoteBuffer for details.
   * */
  uint64_t numElements() const { return shape_.dim(1); }

  uint64_t repeats() const { return shape_.dim(0); }

  DType dtype() const { return dtype_; }

  const RemoteOptions &options() const { return options_; }

  /**
   * The ipu to which this remote is associated.
   * */
  DeviceId ipu() const { return ipu_; }

  std::string handle() const { return str() + ":" + options_.handle(); }

private:
  bool canStoreShape(const Shape &) const final;
  bool canStoreDType(DType) const final;
  DeviceId ipu_;
  DType dtype_;
  Shape shape_;
  RemoteOptions options_;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
