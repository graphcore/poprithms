// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "error.hpp"

#include <set>
#include <sstream>
#include <tuple>

#include <poprithms/program/callstack/copyin.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace program {
namespace callstack {

std::string CopyIns::str() const {
  std::ostringstream oss;
  append(oss);
  return oss.str();
}

TensorIds CopyIns::srcIds() const {
  TensorIds ids;
  ids.reserve(copyIns_.size());
  for (const auto &x : copyIns_) {
    ids.push_back(x.src());
  }
  return ids;
}

TensorIds CopyIns::dstIds() const {
  TensorIds ids;
  ids.reserve(copyIns_.size());
  for (const auto &x : copyIns_) {
    ids.push_back(x.dst());
  }
  return ids;
}

void CopyIn::append(std::ostream &ost) const {
  ost << "(" << src() << "->" << dst();
  if (index_u32() != 0) {
    ost << ":" << index();
  }
  ost << ")";
}

void CopyIns::append(std::ostream &ost) const {
  poprithms::util::append(ost, copyIns_);
}

std::ostream &operator<<(std::ostream &ost, const CopyIn &copyIn) {
  copyIn.append(ost);
  return ost;
}
std::ostream &operator<<(std::ostream &ost, const CopyIns &copyIns) {
  copyIns.append(ost);
  return ost;
}

CopyIns CopyIns::zip(const TensorIds &srcs,
                     const TensorIds &dsts,
                     const CalleeIndex index) {
  return zip(srcs, dsts, CalleeIndices(srcs.size(), index));
}

CopyIns CopyIns::zip(const TensorIds &srcs,
                     const TensorIds &dsts,
                     const CalleeIndices &indices) {
  if (srcs.size() != dsts.size()) {
    std::ostringstream oss;
    oss << "'srcs' and 'dsts' are not the same size in CopyIns::zip: "
        << srcs.size() << " != " << dsts.size();
    throw error(oss.str());
  }
  if (srcs.size() != indices.size()) {
    std::ostringstream oss;
    oss << "'srcs' and 'indices' not same size in CopyIns::zip: "
        << srcs.size() << " != " << indices.size();
    throw error(oss.str());
  }
  std::vector<CopyIn> cins;
  cins.reserve(srcs.size());
  for (uint64_t i = 0; i < srcs.size(); ++i) {
    cins.push_back({srcs[i], dsts[i], indices[i]});
  }
  return CopyIns(std::move(cins));
}

bool CopyIns::destinationsUniqueAtAllIndices() const {
  std::set<std::pair<TensorId, CalleeIndex>> dests;
  for (const auto &x : copyIns_) {
    dests.insert(std::pair<TensorId, CalleeIndex>(x.dst(), x.index()));
  }
  return dests.size() == copyIns_.size();
}

void CopyIns::assertDestinationsUniqueAtAllIndices() const {
  if (!destinationsUniqueAtAllIndices()) {
    throw error("Cannot have a destination in the callee graph with multiple "
                "copy sources");
  }
}

bool CopyIns::sourcesUniqueAtAllIndices() const {
  std::set<std::pair<TensorId, CalleeIndex>> srcs;
  for (const auto &x : copyIns_) {
    srcs.insert(std::pair<TensorId, CalleeIndex>(x.src(), x.index()));
  }
  return srcs.size() == copyIns_.size();
}

bool CopyIns::isDst(CalleeIndex ci, const TensorId &tId) const {
  for (const auto &copyIn : copyIns_) {
    if (copyIn.index() == ci && copyIn.dst() == tId) {
      return true;
    }
  }
  return false;
}

TensorId CopyIns::src(CalleeIndex ci, const TensorId &tId) const {
  TensorIds found;
  for (const auto &copyIn : copyIns_) {
    if (copyIn.index() == ci && copyIn.dst() == tId) {
      found.push_back(copyIn.src());
    }
  }

  if (found.size() != 1) {
    std::ostringstream oss;
    oss << "Expected only one source for TensorId=" << tId
        << " with CalleeIndex=" << ci << ", but there are " << found.size()
        << '.';
    if (found.size() != 0) {
      oss << " This is strange, they are " << found << '.';
    }
    throw error(oss.str());
  }
  return found[0];
}

CopyIns::CopyIns(const std::vector<CopyIn> &cis) {

  // 1) remove duplicates and make nicely ordered:
  std::set<CopyIn> cis_(cis.cbegin(), cis.cend());
  copyIns_ = std::vector<CopyIn>(cis_.cbegin(), cis_.cend());

  // If there is a tensor in a callee graph which has 2 sources, that's a
  // problem as it's probably ambiguous which one gets copied first.
  assertDestinationsUniqueAtAllIndices();
}

} // namespace callstack
} // namespace program
} // namespace poprithms
