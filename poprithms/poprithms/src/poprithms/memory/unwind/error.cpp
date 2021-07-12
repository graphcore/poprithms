// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory/unwind/error.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

poprithms::error::error error(const std::string &what) {
  static const std::string pr("poprithms::memory::unwind");
  return poprithms::error::error(pr, what);
}

} // namespace unwind
} // namespace memory
} // namespace poprithms
