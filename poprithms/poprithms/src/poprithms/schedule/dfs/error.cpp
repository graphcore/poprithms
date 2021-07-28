// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <schedule/dfs/error.hpp>

namespace poprithms {
namespace schedule {
namespace dfs {

namespace {
constexpr const char *const nspace("schedule::dfs");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(error::Code code, const std::string &what) {
  return poprithms::error::error(nspace, code, what);
}

} // namespace dfs
} // namespace schedule
} // namespace poprithms
