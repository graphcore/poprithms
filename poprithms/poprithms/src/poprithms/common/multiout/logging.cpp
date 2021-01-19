// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poprithms/common/multiout/logging.hpp>
#include <poprithms/logging/logging.hpp>

namespace poprithms {
namespace common {
namespace multiout {

poprithms::logging::Logger &log() {
  static poprithms::logging::Logger logger("common::multiout");
  return logger;
}

} // namespace multiout
} // namespace common
} // namespace poprithms
