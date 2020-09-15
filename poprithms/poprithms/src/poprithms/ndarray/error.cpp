// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/ndarray/error.hpp>

namespace poprithms {
namespace ndarray {

poprithms::error::error error(const std::string &what) {
  static const std::string utilStr("ndarray");
  return poprithms::error::error(utilStr, what);
}

} // namespace ndarray
} // namespace poprithms
