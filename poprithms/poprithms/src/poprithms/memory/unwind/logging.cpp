// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poprithms/logging/logging.hpp>
#include <poprithms/memory/unwind/logging.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

poprithms::logging::Logger &log() {
  static poprithms::logging::Logger logger("memory::unwind");
  return logger;
}

} // namespace unwind
} // namespace memory
} // namespace poprithms
