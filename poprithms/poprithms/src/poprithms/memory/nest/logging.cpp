// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/logging/logging.hpp>
#include <poprithms/memory/nest/logging.hpp>

namespace poprithms {
namespace memory {
namespace nest {

poprithms::logging::Logger &log() {
  static poprithms::logging::Logger logger("memory::nest");
  return logger;
}

} // namespace nest
} // namespace memory
} // namespace poprithms
