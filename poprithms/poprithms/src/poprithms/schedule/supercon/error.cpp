// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <schedule/supercon/error.hpp>

namespace poprithms {
namespace schedule {
namespace supercon {

namespace {
constexpr const char *const nspace("schedule::supercon");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(uint64_t id, const std::string &what) {
  return poprithms::error::error(nspace, id, what);
}

} // namespace supercon
} // namespace schedule
} // namespace poprithms
