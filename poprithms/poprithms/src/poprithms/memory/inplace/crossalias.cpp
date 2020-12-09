// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/memory/inplace/crossalias.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

void CrossAlias::append(std::ostream &ost) const {
  ost << in() << "->" << out();
  if (isModifying()) {
    ost << "[modifying]";
  } else {
    ost << "[not modifying]";
  }
}

std::ostream &operator<<(std::ostream &ost, const CrossAliases &m) {
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

std::ostream &operator<<(std::ostream &ost, const CrossAlias &ca) {
  ca.append(ost);
  return ost;
}

} // namespace inplace
} // namespace memory
} // namespace poprithms
