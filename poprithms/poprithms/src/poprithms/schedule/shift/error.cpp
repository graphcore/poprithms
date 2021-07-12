// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <schedule/shift/error.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

poprithms::error::error error(const std::string &what) {
  static const std::string shift("schedule::shift");
  return poprithms::error::error(shift, what);
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
