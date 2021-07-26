// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <common/multiout/error.hpp>

namespace poprithms {
namespace common {
namespace multiout {

namespace {
constexpr const char *const nspace("common::multiout");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(uint64_t id, const std::string &what) {
  return poprithms::error::error(nspace, id, what);
}

} // namespace multiout
} // namespace common
} // namespace poprithms
