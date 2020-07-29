// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/logging/logging.hpp>
#include <poprithms/memory/alias/logging.hpp>

namespace poprithms {
namespace memory {
namespace alias {

poprithms::logging::Logger &log() {
  static poprithms::logging::Logger logger("memory::alias");
  return logger;
}

} // namespace alias
} // namespace memory
} // namespace poprithms
