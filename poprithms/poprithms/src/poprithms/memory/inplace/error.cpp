// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory/inplace/error.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

namespace {
constexpr const char *const nspace("memory::inplace");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(error::Code code, const std::string &what) {
  return poprithms::error::error(nspace, code, what);
}

} // namespace inplace
} // namespace memory
} // namespace poprithms
