// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory/chain/error.hpp>

namespace poprithms {
namespace memory {
namespace chain {

namespace {
constexpr const char *const nspace("memory::chain");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(error::Code code, const std::string &what) {
  return poprithms::error::error(nspace, code, what);
}

} // namespace chain
} // namespace memory
} // namespace poprithms
