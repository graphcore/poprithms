// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <ndarray/error.hpp>

namespace poprithms {
namespace ndarray {

namespace {
constexpr const char *const nspace("ndarray");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(uint64_t id, const std::string &what) {
  return poprithms::error::error(nspace, id, what);
}

} // namespace ndarray
} // namespace poprithms
