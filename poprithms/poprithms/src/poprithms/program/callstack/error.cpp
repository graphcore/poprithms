// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <program/callstack/error.hpp>

namespace poprithms {
namespace program {
namespace callstack {

namespace {
constexpr const char *const nspace("program::callstack");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(error::Code code, const std::string &what) {
  return poprithms::error::error(nspace, code, what);
}

} // namespace callstack
} // namespace program
} // namespace poprithms
