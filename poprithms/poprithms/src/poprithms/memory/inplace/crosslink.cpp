// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <typeinfo>

#include <poprithms/memory/inplace/crosslink.hpp>
#include <poprithms/util/copybyclone_impl.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

void CrossLink::append(std::ostream &ost) const {
  ost << in() << "->" << out();
  if (isModifying()) {
    ost << "[modifying]";
  } else if (isAliasing()) {
    ost << "[aliasing]";
  } else {
    ost << "[using]";
  }
}

CrossLink CrossLink::modifies(InIndex i, OutIndex o) {
  return CrossLink(i, o, Type::Modifies);
}

CrossLink CrossLink::pureAliases(InIndex i, OutIndex o) {
  return CrossLink(i, o, Type::PureAliases);
}

std::ostream &operator<<(std::ostream &ost, const CrossLinks &m) {
  ost << '(';
  if (!m.empty()) {
    m[0].append(ost);
  }
  for (uint64_t i = 1; i < m.size(); ++i) {
    ost << ',';
    m[i].append(ost);
  }
  ost << ')';
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const CrossLink &ca) {
  ca.append(ost);
  return ost;
}

} // namespace inplace
} // namespace memory

} // namespace poprithms
