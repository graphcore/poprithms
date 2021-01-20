// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poprithms/logging/logging.hpp>
#include <poprithms/memory/chain/logging.hpp>

namespace poprithms {
namespace memory {
namespace chain {

poprithms::logging::Logger &log() {
  static poprithms::logging::Logger logger("memory::chain");
  return logger;
}

} // namespace chain
} // namespace memory
} // namespace poprithms
