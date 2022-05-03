// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <common/compute/error.hpp>

namespace poprithms {
namespace common {
namespace compute {

namespace {
constexpr const char *const nspace("common::compute");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(error::Code code, const std::string &what) {
  return poprithms::error::error(nspace, code, what);
}

} // namespace compute
} // namespace common
} // namespace poprithms
