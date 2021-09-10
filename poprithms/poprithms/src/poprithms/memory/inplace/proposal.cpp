// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory/inplace/error.hpp>

#include <poprithms/memory/inplace/proposal.hpp>
#include <poprithms/memory/inplace/tensor.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

void Proposal::append(std::ostream &ost) const {
  ost << "(id=" << aliasGateId() << ", index=" << inIndex() << ')';
}

std::ostream &operator<<(std::ostream &ost, const Proposal &p) {
  p.append(ost);
  return ost;
}

Proposals Proposal::open0(const OpIds &ids) {
  Proposals ps;
  ps.reserve(ids.size());
  for (const auto &id : ids) {
    ps.push_back({id, 0});
  }
  return ps;
}

Proposals Proposal::open0(const TensorIds &ids) {
  Proposals ps;
  ps.reserve(ids.size());
  for (const auto &id : ids) {
    ps.push_back({id.opId(), 0});
  }
  return ps;
}

Proposal::Proposal(const Tensor &tid, InIndex index)
    : Proposal(tid.opId(), index) {}

} // namespace inplace
} // namespace memory
} // namespace poprithms
