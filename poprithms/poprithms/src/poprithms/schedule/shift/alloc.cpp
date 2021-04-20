// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>

#include <poprithms/schedule/shift/alloc.hpp>
#include <poprithms/schedule/shift/error.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

void Alloc::insertOp(OpAddress opAddress) {
  auto it = std::lower_bound(ops.begin(), ops.end(), opAddress);
  if (it == ops.end() || opAddress != *it) {
    ops.insert(it, opAddress);
  }
}

void Alloc::appendSerialization(std::ostream &ost) const {
  ost << "{\"address\":" << address << ",\"weight\":";
  weight.appendSerialization(ost);
  ost << '}';
}

std::ostream &operator<<(std::ostream &ost, const Alloc &alloc) {
  ost << alloc.getAddress() << " ops=";
  util::append(ost, alloc.getOps());
  ost << " weight=" << alloc.getWeight();
  return ost;
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
