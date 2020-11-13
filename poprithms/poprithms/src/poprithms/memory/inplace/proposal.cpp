// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/proposal.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

void Proposal::append(std::ostream &ost) const {
  ost << "(id=" << tensorId() << ", t=" << type() << ')';
}

std::ostream &operator<<(std::ostream &ost, const Proposal &p) {
  p.append(ost);
  return ost;
}

void Proposal::assertValidType() const {
  if (type().isOutplace() || type().isNone()) {
    std::ostringstream oss;
    oss << "Invalid Proposal for TensorId " << tensorId()
        << ". Proposals cannot be " << type() << '.';
    throw error(oss.str());
  }
}

} // namespace inplace
} // namespace memory
} // namespace poprithms
