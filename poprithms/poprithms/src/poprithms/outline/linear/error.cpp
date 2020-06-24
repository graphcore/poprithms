// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/outline/linear/error.hpp>

namespace poprithms {
namespace outline {
namespace linear {

poprithms::error::error error(const std::string &what) {
  static const std::string linear("outline::linear");
  return poprithms::error::error(linear, what);
}

} // namespace linear
} // namespace outline
} // namespace poprithms
