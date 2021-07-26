// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <util/error.hpp>

namespace poprithms {
namespace util {

namespace {
constexpr const char *const nspace("util");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(uint64_t id, const std::string &what) {
  return poprithms::error::error(nspace, id, what);
}

} // namespace util
} // namespace poprithms
