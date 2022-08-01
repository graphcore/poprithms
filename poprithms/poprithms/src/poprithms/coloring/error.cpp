// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <outline/linear/error.hpp>

namespace poprithms {
namespace coloring {

namespace {
constexpr const char *const nspace("coloring");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(error::Code code, const std::string &what) {
  return poprithms::error::error(nspace, code, what);
}

} // namespace coloring
} // namespace poprithms
