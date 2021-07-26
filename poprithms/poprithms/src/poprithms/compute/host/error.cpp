// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <compute/host/error.hpp>

namespace poprithms {
namespace compute {
namespace host {

namespace {
constexpr const char *const nspace("compute::host");
}

poprithms::error::error error(const std::string &what) {
  return poprithms::error::error(nspace, what);
}

poprithms::error::error error(uint64_t id, const std::string &what) {
  return poprithms::error::error(nspace, id, what);
}

} // namespace host
} // namespace compute
} // namespace poprithms
