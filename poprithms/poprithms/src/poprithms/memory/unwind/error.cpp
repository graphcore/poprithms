// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory/unwind/error.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

namespace {
constexpr const char *const nspace("memory::unwind");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(error::Code code, const std::string &what) {
  return poprithms::error::error(nspace, code, what);
}

} // namespace unwind
} // namespace memory
} // namespace poprithms
