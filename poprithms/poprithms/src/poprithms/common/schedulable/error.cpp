// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <common/schedulable/error.hpp>

namespace poprithms {
namespace common {
namespace schedulable {

poprithms::error::error error(const std::string &what) {
  static const std::string pr("poprithms::common::schedulable");
  return poprithms::error::error(pr, what);
}

} // namespace schedulable
} // namespace common
} // namespace poprithms
