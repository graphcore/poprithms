// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <schedule/dfs/error.hpp>

namespace poprithms {
namespace schedule {
namespace dfs {

poprithms::error::error error(const std::string &what) {
  static const std::string dfs("schedule::dfs");
  return poprithms::error::error(dfs, what);
}

} // namespace dfs
} // namespace schedule
} // namespace poprithms
