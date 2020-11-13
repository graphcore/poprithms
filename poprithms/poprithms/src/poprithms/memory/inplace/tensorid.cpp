// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <ostream>
#include <sstream>

#include <poprithms/memory/alias/usings.hpp>
#include <poprithms/memory/inplace/tensorid.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

std::ostream &operator<<(std::ostream &ost, const TensorId &id) {
  id.append(ost);
  return ost;
}
std::string TensorId::str() const {
  std::ostringstream oss;
  oss << *this;
  return oss.str();
}

std::ostream &operator<<(std::ostream &ost, const TensorIds &ids) {
  std::vector<std::string> frags;
  frags.reserve(ids.size());
  for (const auto x : ids) {
    frags.push_back(x.str());
  }
  poprithms::util::append(ost, frags);
  return ost;
}

void TensorId::append(std::ostream &ost) const {
  ost << '(' << opId() << ',' << outIndex() << ')';
}

} // namespace inplace
} // namespace memory
} // namespace poprithms
