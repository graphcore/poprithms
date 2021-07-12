// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <compute/host/error.hpp>

namespace poprithms {
namespace compute {
namespace host {

poprithms::error::error error(const std::string &what) {
  static const std::string shift("compute::host");
  return poprithms::error::error(shift, what);
}

} // namespace host
} // namespace compute
} // namespace poprithms
