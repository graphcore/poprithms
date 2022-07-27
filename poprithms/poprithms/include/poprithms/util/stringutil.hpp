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

  enum class Align { Left = 0, Right };

  /**
   * Parameters controlling the geometry of a column.
   * */
  class Parameters {

  public:
    Parameters() = default;

    /**
     * The character which is used to fill the line between the title and the
     * entries.
     * */
    Parameters &delimiter(char c);
    char delimiter() const { return delimiter_; }

    /**
     * How the column should be aligned.
     * */
    Parameters &alignType(Align);
    Align alignType() const { return alignType_; }

    /**
     * The maximum width that an entry in the column can have. Entries which
     * exceed this length will be abridged (the center characters will be
     * removed) if #abridegeToSingleRow is true, else they will run over
     * mutiple lines.
     * */
    Parameters &thresholdWidth(uint64_t t);
    uint64_t thresholdWidth() const { return thresholdWidth_; }

    /**
     * If an entry exceeds #thresholdWidth, then it will either be abridged
     * (center removed) or it will run over multiple rows.
     * */
    bool abridgeToSingleRow() const { return abridgeToSingleRow_; }
    Parameters &abridgeToSingleRow(bool a);

  private:
    char delimiter_{'-'};
    Align alignType_{Align::Left};
    uint64_t thresholdWidth_{100};
    bool abridgeToSingleRow_{false};
  };

  /**
   * A column in an aligned table.
   *
   * \param title The title of the column
   *
   * \param entries All of the rows (excluding the title) of the column
   *
   * \param parameters The geometry of the column
   * */

  StringColumn(const std::string &title,
               const std::vector<std::string> &entries,
               const Parameters &);

  StringColumn(const std::string &title,
               const std::vector<std::string> &entries,
               char delimiter,
               Align alignType,
               uint64_t thresholdWidth,
               bool abridgeToSingleRow);

  const std::string &title() const { return title_; }
  std::string entry(uint64_t i) const { return entries_[i]; }
  const std::vector<std::string> &entries() const { return entries_; }
  char delimiter() const { return delimiter_; }

  /**
   * The maximum width, over title and all entries. This will never exceed the
   * constructor argument, #thresholdWidth.
   * */
  uint64_t width() const { return width_; }
  uint64_t nEntries() const { return entries_.size(); }
  Align align() const { return align_; }

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

  /**
   * If there are 2 entries which are different, then return false. Otherwise
   * return true.
   * */
  bool entriesAllIdentical() const;

private:
  std::string title_;
  std::vector<std::string> entries_;
  char delimiter_ = '-';
  uint64_t width_;
  Align align_;
};

using StringColumns = std::vector<StringColumn>;

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
 *
 * */
std::string alignedColumns(const StringColumns &);

/**
 * An extension to the method #alignedColumns which omits all columns which
 * have the same value (string) in every row. That is, 'mono-columns' are
 * removed from the main table. For example, instead of the string
 *
 * """
 *     title0 title1 title2 title3
 *     ------ ------ ------ ------
 *     a      d      f      g
 *     b      e      f      g
 *     c      e      f      g
 * """
 *
 * The string returned by this method is,
 *
 * """
 *     Entry title2 title3
 *     ----- ------ ------
 *     *     f      g
 *
 *     title0 title1
 *     ------ ------
 *     a      d
 *     b      e
 *     c      e
 * """
 *
 * The first table summarizes the mono-columns, and the second (main) table
 * contains all the columns which contain different values.
 *
 * \param rowThreshold If the number of rows is below this threshold, then
 *        mono-columns are not removed from the main table.
 *
 * */
std::string alignedColumnsWithMonoColumnsAbridged(const StringColumns &,
                                                  uint64_t rowThreshold);

/**
 * Some variadic string sugar
 * */
namespace cat {
inline void append(std::ostream &) {}
template <class Arg0, class... Args>
void append(std::ostream &oss, Arg0 &&arg0, Args &&...args) {
  oss << arg0;
  return append(oss, args...);
}
template <class... Args> std::string strcat(Args &&...args) {
  std::ostringstream oss;
  append(oss, args...);
  return oss.str();
}
} // namespace cat

} // namespace util
} // namespace poprithms

#endif
