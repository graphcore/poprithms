// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/logging/logging.hpp>
#include <poprithms/schedule/supercon/logging.hpp>

namespace poprithms {
namespace schedule {
namespace supercon {

poprithms::logging::Logger &log() {
  static poprithms::logging::Logger logger("supercon");
  return logger;
}

} // namespace supercon
} // namespace schedule
} // namespace poprithms
