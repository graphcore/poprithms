// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <schedule/shift/error.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

namespace {
constexpr const char *const nspace("schedule::shift");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(uint64_t id, const std::string &what) {
  return poprithms::error::error(nspace, id, what);
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
