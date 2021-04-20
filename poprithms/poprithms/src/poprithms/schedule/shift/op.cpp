// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <array>
#include <iomanip>

#include <poprithms/schedule/shift/error.hpp>
#include <poprithms/schedule/shift/op.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

std::ostream &operator<<(std::ostream &ost, const Op &op) {
  op.append(ost);
  return ost;
}

template <typename C, typename T> void insertUniqueAscending(C &vals, T nxt) {
  auto it = std::lower_bound(vals.begin(), vals.end(), nxt);
  if (it == vals.end() || nxt != *it) {
    vals.insert(it, nxt);
  }
}

void Op::append(std::ostream &ost) const { ost << debugString; }

void Op::insertOut(OpAddress out) { insertUniqueAscending(outs, out); }

void Op::insertIn(OpAddress i) { insertUniqueAscending(ins, i); }

void Op::insertAlloc(AllocAddress aa) { insertUniqueAscending(allocs, aa); }

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

} // namespace shift
} // namespace schedule
} // namespace poprithms
