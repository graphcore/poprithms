// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <schedule/vanilla/error.hpp>

namespace poprithms {
namespace schedule {
namespace vanilla {

poprithms::error::error error(const std::string &what) {
  static const std::string vanillaPrefix("schedule::vanilla");
  return poprithms::error::error(vanillaPrefix, what);
}

} // namespace vanilla
} // namespace schedule
} // namespace poprithms
