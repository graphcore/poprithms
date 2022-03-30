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

InIndex CopyIns::inIndex(CalleeIndex ci, const TensorId &inCallee) const {

  for (uint64_t i = 0; i < copyIns_.size(); ++i) {
    if (copyIns_[i].index() == ci) {
      if (copyIns_[i].dst() == inCallee) {
        return i;
      }
    }
  }

  std::ostringstream oss;
  oss << "Failed to find an index for CalleeIndex=" << ci << " for which "
      << inCallee << " is the destination. ";
  throw error(oss.str());
}

CalleeTensorIds CopyIns::indexedDsts(const InIndices &inIndices) const {
  CalleeTensorIds indexedTensors;
  indexedTensors.reserve(inIndices.size());
  for (auto i : inIndices) {
    const auto &copyIn_ = copyIns_.at(i.get());
    indexedTensors.push_back({copyIn_.dst(), copyIn_.index()});
  }
  return indexedTensors;
}

TensorIds CopyIns::dsts(const InIndices &inIndices) const {
  TensorIds ts;
  ts.reserve(inIndices.size());
  for (auto i : inIndices) {
    const auto &copyIn_ = copyIns_.at(i.get());
    ts.push_back(copyIn_.dst());
  }
  return ts;
}

TensorIds CopyIns::dsts(CalleeIndex ci, const TensorId &inCaller) const {
  TensorIds dsts_;
  for (auto &&cIn : copyIns()) {
    if (cIn.index() == ci && cIn.src() == inCaller) {
      dsts_.push_back(cIn.dst());
    }
  }

  return dsts_;
}

TensorIds CopyIns::srcs(CalleeIndex ci) const {
  TensorIds ids;
  for (const auto &copyIn : copyIns_) {
    if (copyIn.index() == ci) {
      ids.push_back(copyIn.src());
    }
  }
  return ids;
}

TensorIds CopyIns::dsts(CalleeIndex ci) const {
  TensorIds ids;
  for (const auto &copyIn : copyIns_) {
    if (copyIn.index() == ci) {
      ids.push_back(copyIn.dst());
    }
  }
  return ids;
}

InIndices CopyIns::indicesOfSrc(CalleeIndex ci,
                                const TensorId &inCaller) const {
  InIndices is;
  for (InIndex i = 0; i < nInTensors(); ++i) {
    if (calleeIndex(i) == ci && src(i) == inCaller) {
      is.push_back(i);
    }
  }

  return is;
}

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

std::vector<CopyIn> CopyIns::zip(const TensorIds &srcs,
                                 const TensorIds &dsts,
                                 const CalleeIndex index) {
  return zip(srcs, dsts, CalleeIndices(srcs.size(), index));
}

std::vector<CopyIn> CopyIns::zip(const TensorIds &srcs,
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

  return cins;
}

bool CopyIns::destinationsUniqueAtAllIndices() const {
  std::set<CalleeTensorId> dests;
  for (const auto &x : copyIns_) {
    dests.insert(CalleeTensorId(x.dst(), x.index()));
  }
  return dests.size() == copyIns_.size();
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

CopyIns::CopyIns(const std::vector<CopyIn> &cis) : copyIns_(cis) {

  if (!destinationsUniqueAtAllIndices()) {
    throw error("Cannot have a destination in the callee graph with multiple "
                "copy sources");
  }
}

} // namespace callstack
} // namespace program
} // namespace poprithms
