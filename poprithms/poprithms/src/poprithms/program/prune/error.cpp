// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <program/prune/error.hpp>

namespace poprithms {
namespace program {
namespace prune {

namespace {
constexpr const char *const nspace("program::prune");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(error::Code code, const std::string &what) {
  return poprithms::error::error(nspace, code, what);
}

} // namespace prune
} // namespace program
} // namespace poprithms
