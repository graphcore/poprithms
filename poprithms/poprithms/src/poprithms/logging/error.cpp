// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <logging/error.hpp>

namespace poprithms {
namespace logging {

namespace {
constexpr const char *const nspace("logging");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(error::Code code, const std::string &what) {
  return poprithms::error::error(nspace, code, what);
}

} // namespace logging
} // namespace poprithms
