// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/schedule/anneal/error.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

poprithms::error::error error(const std::string &what) {
  static const std::string anneal("schedule::anneal");
  return poprithms::error::error(anneal, what);
}

} // namespace anneal
} // namespace schedule
} // namespace poprithms
