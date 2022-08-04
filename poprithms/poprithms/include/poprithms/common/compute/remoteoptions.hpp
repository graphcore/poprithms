// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef POPRITHMS_COMMON_COMPUTE_REMOTEOPTIONS_HPP
#define POPRITHMS_COMMON_COMPUTE_REMOTEOPTIONS_HPP

#include <poprithms/common/compute/device.hpp>

namespace poprithms {
namespace common {
namespace compute {
/**
 * Options for remote devices, these correspond 1:1 with poplar's remote
 * buffer options (see poplar::Graph::addRemoteBuffer for details).
 * */
struct RemoteOptions {
public:
  RemoteOptions() = default;

  /**
   * Option getters
   * */
  bool rearrangeOnHost() const { return rearrangeOnHost_; }
  bool optimizeMemory() const { return optimizeMemory_; }
  std::string handle() const { return handle_; }

  /**
   * Option setters
   * */
  RemoteOptions &rearrangeOnHost(bool);
  RemoteOptions &optimizeMemory(bool);
  RemoteOptions &handle(const std::string &);

  bool operator==(const RemoteOptions &ro) const { return t() == ro.t(); }
  bool operator<(const RemoteOptions &ro) const { return t() < ro.t(); }

private:
  std::tuple<bool, bool, std::string> t() const {
    return {rearrangeOnHost_, optimizeMemory_, handle_};
  }
  bool rearrangeOnHost_{false};
  bool optimizeMemory_{false};
  std::string handle_;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
