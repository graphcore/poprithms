// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <common/schedulable/error.hpp>

namespace poprithms {
namespace common {
namespace schedulable {

namespace {
constexpr const char *const nspace("common::schedulable");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(uint64_t id, const std::string &what) {
  return poprithms::error::error(nspace, id, what);
}

} // namespace schedulable
} // namespace common
} // namespace poprithms
