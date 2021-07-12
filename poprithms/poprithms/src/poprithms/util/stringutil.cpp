// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <numeric>
#include <ostream>
#include <sstream>
#include <vector>

#include <poprithms/util/error.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace util {

StringColumn::StringColumn(const std::string &t,
                           const std::vector<std::string> &es,
                           char d,
                           Align align)
    : title_(t), entries_(es), delimiter_(d), align_(align) {

  // Set the maximum width:
  uint64_t w = title_.size();
  for (const auto &s : entries_) {
    w = std::max<size_t>(w, s.size());
  }
  width_ = w;
}

std::string lowercase(const std::string &x) {
  auto lower = x;
  std::transform(lower.begin(),
                 lower.end(),
                 lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return lower;
}

std::string spaceString(uint64_t target, const std::string &ts) {
  uint64_t taken = ts.size();
  if (taken > target) {
    return std::string(" ");
  }
  return std::string(target - taken + 1, ' ');
}

namespace {

std::string
padded(const std::string &x, uint64_t wd, StringColumn::Align align) {
  const std::string space_(wd > x.size() ? wd - x.size() : 0, ' ');
  switch (align) {
  case StringColumn::Align::Left: {
    return x + space_;
  }
  case StringColumn::Align::Right: {
    return space_ + x + " ";
  }
  }

  throw error("Unhandled alignment case in.");
}

} // namespace

std::string alignedColumns(const std::vector<StringColumn> &scs) {

  if (scs.empty()) {
    return "";
  }

  for (const auto &x : scs) {
    if (scs[0].nEntries() != x.nEntries()) {
      throw error("Entries in alignedColumns must all be of same size");
    }
  }

  std::ostringstream oss;
  for (const auto &sc : scs) {
    oss << padded(sc.title(), sc.width() + 1, sc.align());
  }
  oss << '\n';
  for (const auto &sc : scs) {
    oss << padded(std::string(sc.title().size(), sc.delimiter()),
                  sc.width() + 1,
                  sc.align());
  }

  for (uint64_t i = 0; i < scs[0].nEntries(); ++i) {
    oss << '\n';
    for (const auto &sc : scs) {
      oss << padded(sc.entry(i), sc.width() + 1, sc.align());
    }
  }

  auto x = oss.str();
  return x;
}

} // namespace util
} // namespace poprithms
