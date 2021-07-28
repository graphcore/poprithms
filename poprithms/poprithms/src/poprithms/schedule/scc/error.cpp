// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <schedule/scc/error.hpp>

namespace poprithms {
namespace schedule {
namespace scc {

namespace {
constexpr const char *const nspace("schedule::scc");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(error::Code code, const std::string &what) {
  return poprithms::error::error(nspace, code, what);
}

} // namespace scc
} // namespace schedule
} // namespace poprithms
