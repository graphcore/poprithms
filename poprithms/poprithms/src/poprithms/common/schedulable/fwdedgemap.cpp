// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <poprithms/common/schedulable/fwdedgemap.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace common {
namespace schedulable {

FwdEdgeMap::FwdEdgeMap(const OpIds &opIds) {
  fwdEdgesCompact_.resize(opIds.size());
  for (auto id : opIds) {
    toCompact_.insert(/*hint = */ toCompact_.end(),
                      {id, fromCompact_.size()});
    fromCompact_.push_back(id);
  }
}

std::ostream &operator<<(std::ostream &ost, const FwdEdgeMap &fem) {
  fem.append(ost);
  return ost;
}

void FwdEdgeMap::append(std::ostream &ost) const {
  ost << "from compact" << '\n';
  ost << "------------" << '\n';
  for (uint64_t i = 0; i < fromCompact_.size(); ++i) {
    ost << ' ' << i << " --> " << fromCompact_[i] << '\n';
  }

  ost << "Compact edges" << '\n';
  ost << "-------------" << '\n';
  for (uint64_t i = 0; i < fwdEdgesCompact_.size(); ++i) {
    ost << ' ' << i << " --> ";
    util::append(ost, fwdEdgesCompact_[i]);
    ost << '\n';
  }
}

OpIds FwdEdgeMap::unpacked(const std::vector<uint64_t> &s_u64) const {
  OpIds f;
  f.reserve(s_u64.size());
  for (auto v : s_u64) {
    f.push_back(fromCompact_[v]);
  }
  return f;
}

} // namespace schedulable
} // namespace common
} // namespace poprithms
