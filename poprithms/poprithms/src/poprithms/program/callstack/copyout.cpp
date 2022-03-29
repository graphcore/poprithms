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

TensorIds CopyOuts::outSources(OutIndex o) const {
  TensorIds outs_;
  for (auto ot : outs.at(o.get())) {
    if (!ot.has_value()) {
      throw error("An optional tensor for output index #" +
                  std::to_string(o.get()) + "  is not set. ");
    }
    outs_.push_back(ot.value());
  }
  return outs_;
}

std::ostream &operator<<(std::ostream &ost, const CopyOuts &co) {
  OptionalTensorId foo;
  co.append(ost);
  return ost;
}

namespace {
std::vector<OptionalTensorIds>
getOptionals(const std::vector<TensorIds> &outs_) {
  std::vector<OptionalTensorIds> outs;
  outs.reserve(outs_.size());
  for (const auto &x : outs_) {
    outs.push_back({});
    for (const auto &y : x) {
      outs.back().push_back(y);
    }
  }
  return outs;
}

template <typename T>
void verifyRectangle(const std::vector<std::vector<T>> &outs_) {
  if (!outs_.empty()) {
    for (const auto &o : outs_) {
      if (o.size() != outs_[0].size()) {
        std::ostringstream oss;
        oss << "Invalid CopyOuts. "
            << "Must have the same number of callees at each OutIndex. "
            << "At OutIndex=0, there are " << outs_[0].size()
            << " callees, while at the current OutIndex there are "
            << o.size() << '.';
        throw error(oss.str());
      }
    }
  }
}
} // namespace

CopyOuts::CopyOuts(const std::vector<OptionalTensorIds> &otis,
                   bool checkRectangle) {
  if (checkRectangle) {
    verifyRectangle(otis);
  }
  outs = otis;
}

CopyOuts CopyOuts::fromOptionals(const std::vector<OptionalTensorIds> &otis) {
  return CopyOuts(otis, true);
}

// [outIndex][calleeIndex].
CopyOuts::CopyOuts(const std::vector<TensorIds> &outs_)
    : CopyOuts(getOptionals(outs_), true) {}

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

void CopyOuts::assertValidOutIndex(OutIndex o) const {
  if (o.get() >= nOutTensors()) {
    std::ostringstream oss;
    oss << "Invalid OutIndex=" << o << " with only " << nOutTensors()
        << " outputs.";
    throw error(oss.str());
  }
}

void CopyOuts::assertValidCalleeIndex(CalleeIndex c) const {
  if (c.get() >= nCallees()) {
    std::ostringstream oss;
    oss << "Invalid CalleeIndex=" << c << " with only " << nCallees()
        << " callees.";
    throw error(oss.str());
  }
}

bool CopyOuts::hasValue(OutIndex o, CalleeIndex ci) const {
  assertValidOutIndex(o);
  assertValidCalleeIndex(ci);
  return outs.at(o.get()).at(ci.get()).has_value();
}

TensorId CopyOuts::outSource(OutIndex o, CalleeIndex c) const {

  assertValidOutIndex(o);
  assertValidCalleeIndex(c);

  auto opt = outs.at(o.get()).at(c.get());

  if (!opt.has_value()) {
    throw error("The optional output at index #" + std::to_string(o.get()) +
                " for callee #" + std::to_string(c.get()) + " is not set.");
  }
  return opt.value();
}

TensorIds CopyOuts::outSources(CalleeIndex c) const {
  TensorIds ts;
  ts.reserve(nOutTensors());
  for (const auto &x : outs) {
    auto opt = x.at(c.get());
    if (!opt.has_value()) {
      std::ostringstream oss;
      oss << "The optional output at this output index for callee #" << c
          << " is not set.";
      throw error(oss.str());
    }
    ts.push_back(opt.value());
  }
  return ts;
}

TensorIds CopyOuts::outSources(CalleeIndex c, const OutIndices &ois) const {
  TensorIds ts;
  ts.reserve(ois.size());
  for (auto o : ois) {
    ts.push_back(outSource(o, c));
  }
  return ts;
}

bool CopyOuts::isSource(CalleeIndex calleeIndex, const TensorId &tId) const {
  for (OutIndex o = 0; o < nOutTensors(); ++o) {
    if (outs[o.get()].at(calleeIndex.get()) == tId) {
      return true;
    }
  }
  return false;
}

OutIndex CopyOuts::outIndex(CalleeIndex calleeIndex,
                            const TensorId &tId) const {
  for (OutIndex o = 0; o < nOutTensors(); ++o) {
    if (outs[o.get()].at(calleeIndex.get()) == tId) {
      return o;
    }
  }

  throw error("No output " + tId.str() + " for callee index " +
              std::to_string(calleeIndex.get()));
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
