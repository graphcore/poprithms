// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <schedule/connectedcomponents/error.hpp>

namespace poprithms {
namespace schedule {
namespace connectedcomponents {

poprithms::error::error error(const std::string &what) {
  static const std::string connectedcomponents(
      "schedule::connectedcomponents");
  return poprithms::error::error(connectedcomponents, what);
}

} // namespace connectedcomponents
} // namespace schedule
} // namespace poprithms
