// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/schedule/transitiveclosure/error.hpp>

namespace poprithms {
namespace schedule {
namespace transitiveclosure {

poprithms::error::error error(const std::string &what) {
  static const std::string transitiveclosure("schedule::transitiveclosure");
  return poprithms::error::error(transitiveclosure, what);
}

} // namespace transitiveclosure
} // namespace schedule
} // namespace poprithms
