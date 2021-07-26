// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <outline/linear/error.hpp>

namespace poprithms {
namespace outline {
namespace linear {

namespace {
constexpr const char *const nspace("outline::linear");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(uint64_t id, const std::string &what) {
  return poprithms::error::error(nspace, id, what);
}

} // namespace linear
} // namespace outline
} // namespace poprithms
