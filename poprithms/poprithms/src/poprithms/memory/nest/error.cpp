// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/memory/nest/error.hpp>

namespace poprithms {
namespace memory {
namespace nest {

poprithms::error::error error(const std::string &what) {
  static const std::string memory("memory::nest");
  return poprithms::error::error(memory, what);
}

} // namespace nest
} // namespace memory
} // namespace poprithms
