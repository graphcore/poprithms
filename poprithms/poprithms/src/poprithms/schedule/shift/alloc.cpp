// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>

#include <boost/functional/hash.hpp>

#include <schedule/shift/error.hpp>

#include <poprithms/schedule/shift/alloc.hpp>
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

size_t Alloc::hash() const {
  size_t hash = 0u;
  boost::hash_combine(hash, getTuple());
  return hash;
}

size_t hash_value(const Alloc &a) { return a.hash(); }

} // namespace shift
} // namespace schedule
} // namespace poprithms
