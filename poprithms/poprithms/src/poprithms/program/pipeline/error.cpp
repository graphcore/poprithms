// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <program/pipeline/error.hpp>

namespace poprithms {
namespace program {
namespace pipeline {

namespace {
constexpr const char *const nspace("program::pipeline");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(error::Code code, const std::string &what) {
  return poprithms::error::error(nspace, code, what);
}

} // namespace pipeline
} // namespace program
} // namespace poprithms
