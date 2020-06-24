// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/util/error.hpp>

namespace poprithms {
namespace util {

poprithms::error::error error(const std::string &what) {
  static const std::string utilStr("util");
  return poprithms::error::error(utilStr, what);
}

} // namespace util
} // namespace poprithms
