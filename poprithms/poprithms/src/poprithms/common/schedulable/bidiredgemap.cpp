// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>

#include <common/schedulable/bidiredgemap.hpp>

namespace poprithms {
namespace common {
namespace schedulable {

OpIds BiDirEdgeMap::bwdEdges(OpId id) const {
  auto found = bwds.find(id);
  if (found == bwds.cend()) {
    return {};
  }
  return found->second;
}

OpIds BiDirEdgeMap::fwdEdges(OpId id) const {
  auto found = fwds.find(id);
  if (found == fwds.cend()) {
    return {};
  }
  return found->second;
}

namespace {
template <typename M>
std::vector<std::pair<OpId, OpId>> getPairs(const M &m) {
  std::vector<std::pair<OpId, OpId>> pairs;
  for (const auto &[x, ys] : m) {
    for (auto y : ys) {
      pairs.push_back({x, y});
    }
  }

  return pairs;
}
} // namespace

std::vector<std::pair<OpId, OpId>> BiDirEdgeMap::bwdEdges() const {
  return getPairs(bwds);
}

std::vector<std::pair<OpId, OpId>> BiDirEdgeMap::fwdEdges() const {
  return getPairs(fwds);
}

void BiDirEdgeMap::insert(OpId f, OpId t) {

  {
    auto found = fwds.find(f);
    if (found == fwds.cend()) {
      fwds.insert({f, {t}});
    } else {
      // If edge already present, do nothing and return.
      auto &v = found->second;
      if (std::find(v.cbegin(), v.cend(), t) != v.cend()) {
        return;
      }
      v.push_back(t);
    }
  }
  {
    auto found = bwds.find(t);
    if (found == bwds.cend()) {
      bwds.insert({t, {f}});
    } else {
      found->second.push_back(f);
    }
  }
}

} // namespace schedulable
} // namespace common
} // namespace poprithms
