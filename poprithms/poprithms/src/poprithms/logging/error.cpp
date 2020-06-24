// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/logging/error.hpp>

namespace poprithms {
namespace logging {

poprithms::error::error error(const std::string &what) {
  static const std::string util("logging");
  return poprithms::error::error(util, what);
}

} // namespace logging
} // namespace poprithms
