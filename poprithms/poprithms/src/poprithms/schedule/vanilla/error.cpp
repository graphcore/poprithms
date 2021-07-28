// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <schedule/vanilla/error.hpp>

namespace poprithms {
namespace schedule {
namespace vanilla {

namespace {
constexpr const char *const nspace("schedule::vanilla");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(error::Code code, const std::string &what) {
  return poprithms::error::error(nspace, code, what);
}

} // namespace vanilla
} // namespace schedule
} // namespace poprithms
