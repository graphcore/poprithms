// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef POPRITHMS_COMMON_COMPUTE_COPYOPTIONS_HPP
#define POPRITHMS_COMMON_COMPUTE_COPYOPTIONS_HPP

#include <poprithms/common/compute/device.hpp>

namespace poprithms {
namespace common {
namespace compute {
/**
 * Options for ops which copy between host and ipu.
 * */
class CopyBetweenHostAndIpuOptions {

public:
  CopyBetweenHostAndIpuOptions()  = default;
  ~CopyBetweenHostAndIpuOptions() = default;

  uint64_t bufferingDepth() const { return bufferingDepth_; }
  CopyBetweenHostAndIpuOptions &bufferingDepth(uint64_t v) {
    bufferingDepth_ = v;
    return *this;
  }

  bool operator==(const CopyBetweenHostAndIpuOptions &rhs) const {
    return t() == rhs.t();
  }

  bool operator<(const CopyBetweenHostAndIpuOptions &rhs) const {
    return t() < rhs.t();
  }

private:
  std::tuple<uint64_t> t() const { return bufferingDepth_; }
  uint64_t bufferingDepth_ = 1ull;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
