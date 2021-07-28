// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory/nest/error.hpp>

namespace poprithms {
namespace memory {
namespace nest {

namespace {
constexpr const char *const nspace("memory::nest");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(error::Code code, const std::string &what) {
  return poprithms::error::error(nspace, code, what);
}

} // namespace nest
} // namespace memory
} // namespace poprithms
