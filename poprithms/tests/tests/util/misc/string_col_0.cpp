// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <iostream>
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/util/stringutil.hpp>

namespace {
using namespace poprithms::util;

uint64_t count(const std::string &s, const std::string &sub) {
  uint64_t n{0};
  auto found = s.find(sub);
  while (found != std::string::npos) {
    ++n;
    found = s.find(sub, found + 1);
  }
  return n;
}

// col0  col1  col2  col3  col4   col5    col6     col7      col8       col9
// ----  ----  ----  ----  ----   ----    ----     ----      ----       ----
//          a     a     a     a      a       a        a         a          a
//          b     b     b     b      b       b        b         b          b
//          .    ..   ...  0...  0...9  01...9  01...89  012...89  012...789
//          .    ..   ...  a...  abcde   abcde    abcde     abcde      abcde
//          d     d     d     d      d       d        d         d          d
//          e     e     e     e      e       e        e         e          e

void test0() {

  std::vector<StringColumn> cols;
  for (uint64_t i = 0; i < 10; ++i) {
    const uint64_t abridgeThresholdWidth = i;
    cols.push_back({"col" + std::to_string(i),
                    {"a", "b", "0123456789", "abcde", "d", "e"},
                    '-',
                    StringColumn::Align::Right,
                    abridgeThresholdWidth,
                    /* abridge to single row: */
                    true});
  }

  auto x = alignedColumns(cols);

  // Checking for this row:
  //
  //          .    ..   ...  0...  0...9  01...9  01...89  012...89  012...789
  //
  // which is the one with abbreviated columns.

  if (count(x, "  .    ..   ...  ") == 0) {
    throw poprithms::test::error(
        "Failed in test of abridgeThresholdWidth, with low thresholds (1)");
  }

  if (count(x, "0...  0...9  01...9  01...89  012...89  012...789") != 1) {
    throw poprithms::test::error(
        "Failed in test of abridgeThresholdWidth, with high thresholds (2)");
  }
}

void test1() {

  auto x = alignedColumns(
      {{"col0", {"asdf", "f"}, {}}, {"col1", {"a", "bumble"}, {}}});

  if (count(x, " \n") != 0) {
    throw poprithms::test::error(
        "space before new line, should have been removed");
  }
}

void testSplitRows0() {

  StringColumn col0("col0",
                    {"short", "0123456789abcdefghijkABCDEF"},
                    '+',
                    StringColumn::Align::Left,
                    10,
                    false);

  StringColumn col1("col1",
                    {"0123456789", "beep"},
                    '*',
                    StringColumn::Align::Left,
                    5,
                    false);

  auto ally = alignedColumns({col0, col1});
  std::cout << ally << std::endl;

  std::vector<std::string> lines{"col0        col1",
                                 "++++        ****",
                                 "short       01234",
                                 "            56789",
                                 "0123456789  beep",
                                 "abcdefghij",
                                 "kABCDEF"};

  for (uint64_t l = 0; l < lines.size(); ++l) {
    if (count(ally, lines[l]) != 1) {
      std::ostringstream oss;
      oss << "expected to find the line " << lines[l]
          << " in the summary string, but did not. ";
      throw poprithms::test::error(oss.str());
    }
  }
}

} // namespace

int main() {
  test0();
  test1();
  testSplitRows0();
  return 0;
}
