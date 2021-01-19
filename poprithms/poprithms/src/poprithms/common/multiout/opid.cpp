// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <ostream>
#include <sstream>

#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace common {
namespace multiout {

std::ostream &operator<<(std::ostream &ost, const OpIds &opIds) {
  poprithms::util::append(ost, opIds);
  return ost;
}

} // namespace multiout
} // namespace common
} // namespace poprithms
