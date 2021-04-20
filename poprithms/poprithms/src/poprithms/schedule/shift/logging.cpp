// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poprithms/logging/logging.hpp>
#include <poprithms/schedule/shift/logging.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

poprithms::logging::Logger &log() {
  static poprithms::logging::Logger logger("shift");
  return logger;
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
