// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/schedule/anneal/error.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

poprithms::util::error error(const std::string &what) {
  static const std::string anneal("schedule::anneal");
  return poprithms::util::error(anneal, what);
}

} // namespace anneal
} // namespace schedule
} // namespace poprithms
