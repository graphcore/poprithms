// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <schedule/scc/error.hpp>

namespace poprithms {
namespace schedule {
namespace scc {

poprithms::error::error error(const std::string &what) {
  static const std::string scc("schedule::scc");
  return poprithms::error::error(scc, what);
}

} // namespace scc
} // namespace schedule
} // namespace poprithms
