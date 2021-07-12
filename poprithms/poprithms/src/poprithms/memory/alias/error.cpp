// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <schedule/dfs/error.hpp>

namespace poprithms {
namespace memory {
namespace alias {

poprithms::error::error error(const std::string &what) {
  static const std::string dfs("memory::alias");
  return poprithms::error::error(dfs, what);
}

} // namespace alias
} // namespace memory
} // namespace poprithms
