// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../include/boolimpl.hpp"

#include <poprithms/ndarray/dtype.hpp>

namespace poprithms {
namespace ndarray {
template <> DType get<poprithms::compute::host::BoolImpl>() {
  return DType::Boolean;
}
} // namespace ndarray
} // namespace poprithms

namespace poprithms {
namespace compute {
namespace host {

std::ostream &operator<<(std::ostream &ost, BoolImpl b) {
  ost << b.v;
  return ost;
}

} // namespace host
} // namespace compute
} // namespace poprithms
