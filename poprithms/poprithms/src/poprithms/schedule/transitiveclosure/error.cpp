// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <schedule/transitiveclosure/error.hpp>

namespace poprithms {
namespace schedule {
namespace transitiveclosure {

namespace {
constexpr const char *const nspace("schedule::transitiveclosure");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(error::Code code, const std::string &what) {
  return poprithms::error::error(nspace, code, what);
}

} // namespace transitiveclosure
} // namespace schedule
} // namespace poprithms
