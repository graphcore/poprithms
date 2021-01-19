// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poprithms/common/multiout/error.hpp>

namespace poprithms {
namespace common {
namespace multiout {

poprithms::error::error error(const std::string &what) {
  static const std::string pr("poprithms::common::multiout");
  return poprithms::error::error(pr, what);
}

} // namespace multiout
} // namespace common
} // namespace poprithms
