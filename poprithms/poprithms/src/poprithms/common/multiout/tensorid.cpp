// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <ostream>
#include <sstream>

#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace common {
namespace multiout {

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
  util::append(ost, ids);
  return ost;
}
void TensorId::append(std::ostream &ost) const {
  ost << "(op=" << opId();
  if (outIndex() != 0) {
    ost << ",out=" << outIndex();
  }
  ost << ')';
}

} // namespace multiout
} // namespace common
} // namespace poprithms
