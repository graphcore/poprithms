// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/logging/logging.hpp>
#include <poprithms/memory/inplace/logging.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

poprithms::logging::Logger &log() {
  static poprithms::logging::Logger logger("memory::inplace");
  return logger;
}

} // namespace inplace
} // namespace memory
} // namespace poprithms
