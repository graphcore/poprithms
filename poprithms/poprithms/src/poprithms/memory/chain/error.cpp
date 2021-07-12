// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory/chain/error.hpp>

namespace poprithms {
namespace memory {
namespace chain {

poprithms::error::error error(const std::string &what) {
  static const std::string pr("poprithms::memory::chain");
  return poprithms::error::error(pr, what);
}

} // namespace chain
} // namespace memory
} // namespace poprithms
