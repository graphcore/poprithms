// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/schedule/supercon/error.hpp>

namespace poprithms {
namespace schedule {
namespace supercon {

poprithms::util::error error(const std::string &what) {
  static const std::string supercon("schedule::supercon");
  return poprithms::util::error(supercon, what);
}

} // namespace supercon
} // namespace schedule
} // namespace poprithms
