// Copyright (c) 202 2Graphcore Ltd. All rights reserved.
#include <program/distributed/error.hpp>

namespace poprithms {
namespace program {
namespace distributed {

namespace {
constexpr const char *const nspace("program::distributed");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(error::Code code, const std::string &what) {
  return poprithms::error::error(nspace, code, what);
}

} // namespace distributed
} // namespace program
} // namespace poprithms
