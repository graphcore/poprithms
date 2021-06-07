// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <ostream>
#include <sstream>

#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/optraversal.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace common {
namespace multiout {

std::ostream &operator<<(std::ostream &ost, const OpTraversal &id) {
  id.append(ost);
  return ost;
}
std::string OpTraversal::str() const {
  std::ostringstream oss;
  oss << *this;
  return oss.str();
}
std::ostream &operator<<(std::ostream &ost, const OpTraversals &ids) {
  util::append(ost, ids);
  return ost;
}
void OpTraversal::append(std::ostream &ost) const {
  ost << '(';
  ost << "in=" << inIndex() << ',';
  ost << "op=" << opId();
  ost << ",out=" << outIndex();
  ost << ')';
}

} // namespace multiout
} // namespace common
} // namespace poprithms
