// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory/inplace/error.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

poprithms::error::error error(const std::string &what) {
  static const std::string pr("poprithms::memory::inplace");
  return poprithms::error::error(pr, what);
}

} // namespace inplace
} // namespace memory
} // namespace poprithms
