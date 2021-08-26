// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <numeric>
#include <ostream>
#include <sstream>
#include <util/error.hpp>
#include <vector>

#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace util {

StringColumn::StringColumn(const std::string &t,
                           const std::vector<std::string> &es,
                           char d,
                           Align align,
                           uint64_t abridgeThresholdWidth)
    : title_(t), delimiter_(d), align_(align) {

  entries_.reserve(es.size());
  for (const auto &e : es) {

    // Entries which are at or below the column width threshold get added
    // as-is:
    if (e.size() <= abridgeThresholdWidth) {
      entries_.push_back(e);
    }

    // Entries which have too many characters get abridged to have exactly
    // #abridgeThresholdWidth characters. The abridge string is of the form
    //
    // "first few characters" "..." "last few characters".
    else {

      // l0 + l1 = abrideThresholdWidth.
      const auto l0 = abridgeThresholdWidth / 2;
      const auto l1 = abridgeThresholdWidth - l0;

      // Take the first l0 characters, and set the final one to '.'
      std::string x0{e.cbegin(), e.cbegin() + l0};
      for (uint64_t i = 0; i < 1; ++i) {
        if (i < x0.size()) {
          x0[x0.size() - i - 1] = '.';
        }
      }

      // Take the final l1 characters, and set the first 2 to '.'
      std::string x1{e.cend() - l1, e.cend()};
      for (uint64_t i = 0; i < 2; ++i) {
        if (i < x1.size()) {
          x1[i] = '.';
        }
      }

      // Add the abridged string to the list of column entries.
      entries_.push_back(x0 + x1);
    }
  }

  // Set the maximum width of all columns:
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

// Remove all whitespace appearing directly before a new line character.
std::string withSpaceBeforeNewlineRemoved(const std::string &toSqueeze) {

  // The indices of 'toSqueeze' to retain.
  std::vector<bool> retainMask(toSqueeze.size(), true);

  // 1) find all '\n' in the string toSqueeze.
  std::vector<uint64_t> newLineCharIndices;
  for (uint64_t i = 0; i < toSqueeze.size(); ++i) {
    if (toSqueeze[i] == '\n') {
      newLineCharIndices.push_back(i);
    }
  }

  for (uint64_t s : newLineCharIndices) {
    while (s > 0 && std::isspace(toSqueeze[s - 1]) &&
           toSqueeze[s - 1] != '\n') {
      retainMask[s - 1] = false;
      --s;
    }
  }

  std::vector<char> retained;
  for (uint64_t i = 0; i < toSqueeze.size(); ++i) {
    if (retainMask[i]) {
      retained.push_back(toSqueeze[i]);
    }
  }

  return {retained.cbegin(), retained.cend()};
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

  return withSpaceBeforeNewlineRemoved(oss.str());
}

} // namespace util
} // namespace poprithms
