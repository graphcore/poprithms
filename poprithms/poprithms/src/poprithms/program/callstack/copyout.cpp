// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "error.hpp"

#include <algorithm>
#include <map>
#include <sstream>

#include <poprithms/program/callstack/copyout.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace program {
namespace callstack {

void CopyOuts::append(std::ostream &ost) const {
  if (nCallees() == 1) {
    poprithms::util::append(ost, outSources(CalleeIndex(0)));
  } else {
    std::vector<std::string> outs_;
    for (OutIndex o = 0; o < nOutTensors(); ++o) {
      std::ostringstream oss;
      oss << "OutIndex=" << o << ":" << outs.at(o.get());
      outs_.push_back(oss.str());
    }
    poprithms::util::append(ost, outs_);
  }
}

std::ostream &operator<<(std::ostream &ost, const CopyOuts &co) {
  co.append(ost);
  return ost;
}

// [outIndex][calleeIndex].
CopyOuts::CopyOuts(const std::vector<TensorIds> &outs_) : outs(outs_) {
  if (!outs.empty()) {
    for (const auto &o : outs) {
      if (o.size() != outs[0].size()) {
        std::ostringstream oss;
        oss << "Invalid CopyOuts. "
            << "Must have the same number of callees at each OutIndex. "
            << "At OutIndex=0, there are " << outs[0].size()
            << " callees, while at the current OutIndex there are "
            << o.size() << '.';
        throw error(oss.str());
      }
    }
  }
}

//  [calleeIndex][outIndex]
CopyOuts::CopyOuts(const std::map<CalleeIndex, TensorIds> &m) {

  if (m.empty()) {
    return;
  }

  // Assert that the callee indices are contiguous from 0.
  uint64_t maxCalleeIndex{0};
  uint64_t nOuts{0};
  for (const auto &[ci, tIds] : m) {
    nOuts          = tIds.size();
    maxCalleeIndex = std::max<uint64_t>(maxCalleeIndex, ci.get());
  }

  outs.resize(nOuts);

  if (maxCalleeIndex + 1ull != m.size()) {
    std::ostringstream oss;
    oss << "The callee indices must be contiguous from 0, "
        << "Cannot have a CalleeIndex of " << maxCalleeIndex << " with only "
        << m.size() << " callees. ";
    throw error(oss.str());
  }

  for (CalleeIndex ci = 0; ci < maxCalleeIndex + 1; ++ci) {
    const auto &tIds = m.at(ci);
    if (tIds.size() != nOuts) {
      std::ostringstream oss;
      oss << "All CalleeIndices must have the same number "
          << "of outputs, but at CalleeIndex=" << ci << " there are "
          << tIds.size() << ". Another CalleeIndex has " << nOuts << '.';
      throw error(oss.str());
    }
    for (OutIndex o = 0; o < nOuts; ++o) {
      outs[o.get()].push_back(tIds[o.get()]);
    }
  }
}

TensorId CopyOuts::outSource(OutIndex o, CalleeIndex c) const {
  if (o.get() >= nOutTensors()) {
    std::ostringstream oss;
    oss << "Invalid OutIndex=" << o << " with only " << nOutTensors()
        << " outputs.";
    throw error(oss.str());
  }

  if (c.get() >= nCallees()) {
    std::ostringstream oss;
    oss << "Invalid CalleeIndex=" << c << " with only " << nCallees()
        << " callees.";
    throw error(oss.str());
  }
  return outs.at(o.get()).at(c.get());
}

TensorIds CopyOuts::outSources(CalleeIndex c) const {
  TensorIds ts;
  ts.reserve(nCallees());
  for (const auto &x : outs) {
    ts.push_back(x.at(c.get()));
  }
  return ts;
}

uint64_t CopyOuts::nCallees() const {
  if (outs.empty()) {
    throw error(
        "Cannot determine number of callees, as there are no outputs");
  }
  return outs.at(0).size();
}

std::string CopyOuts::outSourcesString(OutIndex o) const {
  std::ostringstream oss;
  poprithms::util::append(oss, outs.at(o.get()));
  return oss.str();
}

} // namespace callstack
} // namespace program
} // namespace poprithms
