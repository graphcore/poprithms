// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <numeric>
#include <ostream>
#include <sstream>
#include <tuple>
#include <util/error.hpp>
#include <vector>

#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace util {

using Parameters = StringColumn::Parameters;

Parameters &Parameters::delimiter(char c) {
  delimiter_ = c;
  return *this;
}

Parameters &Parameters::alignType(Align a) {
  alignType_ = a;
  return *this;
}

Parameters &Parameters::thresholdWidth(uint64_t t) {
  thresholdWidth_ = t;
  return *this;
}

Parameters &Parameters::abridgeToSingleRow(bool a) {
  abridgeToSingleRow_ = a;
  return *this;
}

StringColumn::StringColumn(const std::string &title,
                           const std::vector<std::string> &entries,
                           const Parameters &p)
    : StringColumn(title,
                   entries,
                   p.delimiter(),
                   p.alignType(),
                   p.thresholdWidth(),
                   p.abridgeToSingleRow()) {}

StringColumn::StringColumn(const std::string &t,
                           const std::vector<std::string> &entries,
                           char delimiter,
                           Align align,
                           uint64_t abridgeThresholdWidth,
                           bool abridgeToSingleRow)
    : title_(t), delimiter_(delimiter), align_(align) {

  if (!abridgeToSingleRow) {
    // if not abridged to a single row, the full entry will be split over
    // multiple rows at a later point.
    entries_ = entries;

  }

  // Abridge to a single row if too many characters:
  else {

    entries_.reserve(entries.size());
    for (const auto &e : entries) {

      // Entries which are at or below the column width threshold get added
      // as-is:
      if (e.size() <= abridgeThresholdWidth) {
        entries_.push_back(e);
      }

      // Entries which have too many characters get abridged to have exactly
      // #abridgeThresholdWidth characters. The center characters are removed,
      // so that the abridged string for "the quick brown fox jumped over the
      // moon" might be "the quick br...ver the moon"
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
  }

  // Set the maximum width of all columns:
  uint64_t w = title_.size();
  for (const auto &s : entries_) {
    w = std::max<size_t>(w, s.size());
  }
  if (!abridgeToSingleRow) {
    w = std::min(abridgeThresholdWidth, w);
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

/// #toPad is a string to pad to a length which is a multiple of #wd,
/// and split into multiple strings of size #wd. The type of padding
/// (before or after) depends on #align.
std::vector<std::string>
padAndSplit(std::string toPad, uint64_t wd, StringColumn::Align align) {

  if (toPad.empty()) {
    return std::vector<std::string>{std::string(wd, ' ')};
  }

  // smallest N s.t. toPad.size() * wd <= N.
  auto N = toPad.size() / wd + (toPad.size() % wd != 0);

  // the remainder (the bit to be padded).
  auto edgeCase = toPad.size() % wd;
  edgeCase      = edgeCase == 0 ? wd : edgeCase;

  const std::string space_(wd - edgeCase, ' ');

  // We perform the padding before doing the splitting.
  switch (align) {
  case StringColumn::Align::Left: {
    toPad = toPad + space_;
    break;
  }
  case StringColumn::Align::Right: {
    toPad = space_ + toPad;
    break;
  }
  default:
    throw error("Unhandled alignment case in.");
  }

  // we'll split 'toPad' into N sub-strings, each exactly wd in length
  std::vector<std::string> split;
  split.reserve(N);
  for (uint64_t n = 0; n < N; ++n) {
    split.push_back(toPad.substr(n * wd, wd));
  }
  return split;
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

bool StringColumn::entriesAllIdentical() const {

  // Impossible to have 2 entries which are different if there are 0 or 1
  // entries only:
  if (entries_.size() < 2) {
    return true;
  }

  // Compare all entries to the first entry. If there is a difference, return
  // false.
  auto &&e0 = entries_[0];
  return std::all_of(entries_.cbegin() + 1,
                     entries_.cend(),
                     [&e0](const auto &e) { return e == e0; });
}

std::string alignedColumnsWithMonoColumnsAbridged(const StringColumns &scs,
                                                  uint64_t rowThreshold) {

  if (scs.empty()) {
    return "";
  }

  auto &&sc0 = scs[0];

  if (rowThreshold < 2) {
    std::ostringstream oss;
    oss << "Row threshold must be at least 2, but it is " << rowThreshold
        << ".";
    throw error(oss.str());
  }

  if (sc0.nEntries() < rowThreshold) {
    return alignedColumns(scs);
  }

  StringColumns nonRepeatingCols;
  StringColumns repeatingCols{StringColumn(
      "Entry", {"*"}, '-', StringColumn::Align::Left, 100, false)};

  for (const auto &sc : scs) {
    if (!sc.entriesAllIdentical() || sc.nEntries() < 2) {
      nonRepeatingCols.push_back(sc);
    } else {
      if (!sc.entry(0).empty()) {
        repeatingCols.push_back(StringColumn(sc.title(),
                                             {sc.entry(0)},
                                             sc.delimiter(),
                                             sc.align(),
                                             100,
                                             false));
      }
    }
  }

  std::ostringstream oss;

  if (repeatingCols.size() > 1) {
    oss << alignedColumns(repeatingCols);
    oss << "\n";
  }

  oss << alignedColumns(nonRepeatingCols);

  return oss.str();
}

std::string alignedColumns(const StringColumns &scs) {

  if (scs.empty()) {
    return "";
  }

  for (const auto &x : scs) {
    if (scs[0].nEntries() != x.nEntries()) {
      throw error("Entries in alignedColumns must all be of same size");
    }
  }

  std::ostringstream oss;

  auto appendNxt =
      [&oss](
          const std::vector<std::tuple<std::string,        // string entry
                                       uint64_t,           // column width
                                       StringColumn::Align // alignment type
                                       >> &cols) {
        std::vector<std::vector<std::string>> colSplits;

        // the maximum (over colums) of the number of rows an entry is split
        // over. If abridgeToSingleRow is true, then this is always 1.
        uint64_t maxnRows{0};

        for (const auto &col : cols) {
          auto ps = padAndSplit(
              std::get<0>(col), std::get<1>(col), std::get<2>(col));
          maxnRows = std::max<uint64_t>(maxnRows, ps.size());
          colSplits.push_back(ps);
        }

        for (auto &ps : colSplits) {
          while (ps.size() < maxnRows) {
            ps.push_back(std::string(ps.back().size(), ' '));
          }
        }

        for (uint64_t r = 0; r < maxnRows; ++r) {
          for (uint64_t c = 0; c < colSplits.size(); ++c) {
            oss << colSplits[c][r] << ' ' << ' ';
          }
          oss << '\n';
        }
      };

  std::vector<std::tuple<std::string, uint64_t, StringColumn::Align>> cols;
  for (const auto &sc : scs) {
    cols.push_back({sc.title(), sc.width(), sc.align()});
  }
  appendNxt(cols);

  cols.clear();
  for (const auto &sc : scs) {
    cols.push_back({std::string(sc.title().size(), sc.delimiter()),
                    sc.width(),
                    sc.align()});
  }
  appendNxt(cols);

  for (uint64_t i = 0; i < scs[0].nEntries(); ++i) {
    cols.clear();
    for (const auto &sc : scs) {
      cols.push_back({sc.entry(i), sc.width(), sc.align()});
    }
    appendNxt(cols);
  }

  return withSpaceBeforeNewlineRemoved(oss.str());
}

} // namespace util
} // namespace poprithms
