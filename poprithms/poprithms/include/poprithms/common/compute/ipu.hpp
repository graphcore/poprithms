// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef POPRITHMS_COMMON_COMPUTE_IPU_HPP
#define POPRITHMS_COMMON_COMPUTE_IPU_HPP

#include <poprithms/common/compute/device.hpp>
#include <poprithms/util/interval.hpp>

namespace poprithms {
namespace common {
namespace compute {

using poprithms::util::Interval;
using poprithms::util::Intervals;

/**
 * A device type which corresponds to a virtual graph in a poplar::Graph. In
 * other words, this corresponds to a subset of ipu tiles.
 * */
class Ipu : public Device {
public:
  /**
   * \param ipuId The id of this ipu (virtual graph).
   *
   * \param tiles The subset of tiles.
   * */
  Ipu(DeviceId ipuId, const Intervals &tiles)
      : Device(ipuId, DeviceType::Ipu), tiles_(tiles) {}

  std::unique_ptr<Device> clone() const final;

  uint64_t nTiles() const { return tiles_.size(); }

  const Intervals &tiles() const { return tiles_; }

private:
  bool canStoreDType(DType) const final;
  bool canStoreShape(const Shape &) const final { return true; }

  Intervals tiles_;
};

} // namespace compute

} // namespace common
} // namespace poprithms

#endif
