// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <schedule/connectedcomponents/error.hpp>

namespace poprithms {
namespace schedule {
namespace connectedcomponents {

namespace {
constexpr const char *const nspace("schedule::connectedcomponents");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(error::Code code, const std::string &what) {
  return poprithms::error::error(nspace, code, what);
}

} // namespace connectedcomponents
} // namespace schedule
} // namespace poprithms
