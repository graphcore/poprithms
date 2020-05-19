// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <array>
#include <iomanip>

#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/op.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/unisort.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

std::ostream &operator<<(std::ostream &ost, const Op &op) {
  op.append(ost);
  return ost;
}

void Op::append(std::ostream &ost) const { ost << debugString; }

void Op::sortAndMakeUnique() {
  ins    = util::unisorted(ins);
  outs   = util::unisorted(outs);
  allocs = util::unisorted(allocs);
}

void Op::appendSerialization(std::ostream &ost) const {

  ost << "{\"address\":" << address << ",\"outs\":[";
  if (nOuts() != 0) {
    ost << getOut(0);
  }
  for (uint64_t i = 1; i < nOuts(); ++i) {
    ost << ',' << getOut(i);
  }
  ost << "],\"allocs\":[";
  if (nAllocs() != 0) {
    ost << getAlloc(0);
  }
  for (uint64_t i = 1; i < nAllocs(); ++i) {
    ost << ',' << getAlloc(i);
  }

  int64_t fwdLinkFragment =
      hasForwardLink() ? static_cast<int64_t>(fwdLink) : -1;

  std::string serialDebugString;
  serialDebugString.reserve(debugString.size());

  // escape characters from cppreference.com/w/cpp/language/escape
  constexpr std::array<char, 11> escaped{
      '\'', '\"', '\?', '\\', '\a', '\b', '\f', '\n', '\r', '\t', '\v'};

  for (auto c : debugString) {
    if (std::find(escaped.cbegin(), escaped.cend(), c) != escaped.cend()) {
      serialDebugString += '\\';
    }
    serialDebugString += c;
  }
  ost << "],\"debugString\":\"" << serialDebugString
      << "\",\"fwdLink\":" << fwdLinkFragment << "}";
}

Op::Op(OpAddress _address_, const std::string &_debugString_)
    : address(_address_), debugString(_debugString_) {}

} // namespace anneal
} // namespace schedule
} // namespace poprithms
