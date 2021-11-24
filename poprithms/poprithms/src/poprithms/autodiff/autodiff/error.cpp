// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <autodiff/autodiff/error.hpp>

namespace poprithms {
namespace autodiff {

poprithms::error::error error(const std::string &what) {
  static const std::string name("autodiff");
  return poprithms::error::error(name, what);
}

} // namespace autodiff
} // namespace poprithms
