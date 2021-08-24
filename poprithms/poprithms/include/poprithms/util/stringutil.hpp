// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_UTIL_STRINGUTIL_HPP
#define POPRITHMS_UTIL_STRINGUTIL_HPP

#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace util {

std::string lowercase(const std::string &);

/**
 * A logging utility for string aligment. This function returns a string of
 * ' ' of length max(1, target - ts.size() + 1)
 * */
std::string spaceString(uint64_t target, const std::string &ts);

/**
 * Define a column of entries in a table, defined by a \a title, some \a
 * entries, and a \a delimiter. When used with \alignedColumns, a column will
 * appear as,
 *
 *  title
 *  -----
 *  entry[0]
 *  entry[1]
 *    .
 *    .
 *    .
 *
 * */
struct StringColumn {

  template <typename T>
  static std::vector<std::string> entriesFromInts(const std::vector<T> &ts) {
    std::vector<std::string> es;
    es.reserve(ts.size());
    for (const auto &x : ts) {
      es.push_back(std::to_string(x));
    }
    return es;
  }

  template <typename T>
  static std::vector<std::string>
  entriesFromVectors(const std::vector<std::vector<T>> &ts) {
    std::vector<std::string> es;
    es.reserve(ts.size());
    for (const auto &x : ts) {
      std::ostringstream oss;
      append(oss, x);
      es.push_back(oss.str());
    }
    return es;
  }

  enum class Align { Left = 0, Right };
  StringColumn(const std::string &title,
               const std::vector<std::string> &entries,
               char delimiter = '-',
               Align          = Align::Left);

  const std::string &title() const { return title_; }
  std::string entry(uint64_t i) const { return entries_[i]; }
  const std::vector<std::string> &entries() const { return entries_; }
  char delimiter() const { return delimiter_; }

  /** The maximum width, over title and all entries*/
  uint64_t width() const { return width_; }
  uint64_t nEntries() const { return entries_.size(); }
  Align align() const { return align_; }

private:
  std::string title_;
  std::vector<std::string> entries_;
  char delimiter_ = '-';
  uint64_t width_;
  Align align_;
};

/**
 * Return a string of aligned columns, for example:
 *
 * alignedColumns({
 *   {"title1", {"e0", "averylongentry"}, '-'},
 *   {"anotherTitle", {"foo", "anotherEntry"}, '+'}
 * });
 *
 * returns
 *
 *     title1         anotherTitle
 *     ------         ++++++++++++
 *     e0             foo
 *     averylongentry anotherEntry
 *
 * */
std::string alignedColumns(const std::vector<StringColumn> &);

} // namespace util
} // namespace poprithms

#endif
