// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory/alias/error.hpp>

namespace poprithms {
namespace memory {
namespace alias {

namespace {
constexpr const char *const nspace("memory::alias");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(error::Code code, const std::string &what) {
  return poprithms::error::error(nspace, code, what);
}

} // namespace alias
} // namespace memory
} // namespace poprithms
