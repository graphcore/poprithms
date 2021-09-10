// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <sstream>

#include <compute/host/include/gridpointhelper.hpp>

#include <poprithms/error/error.hpp>

namespace {

using namespace poprithms::compute::host;

using Row    = GridPointHelper::Row;
using Column = GridPointHelper::Column;

using Rows    = GridPointHelper::Rows;
using Columns = GridPointHelper::Columns;

using Coord  = GridPointHelper::Coord;
using Coords = GridPointHelper::Coords;

std::ostream &operator<<(std::ostream &ost, const Coord &coord) {
  ost << '(' << std::get<0>(coord) << ',' << std::get<1>(coord) << ')';
  return ost;
}
} // namespace

// postponed inclusion, as the above operator<< must appear before it.
#include <poprithms/util/printiter.hpp>

namespace {

template <typename T>
std::ostream &operator<<(std::ostream &ost, const std::vector<T> &t) {
  poprithms::util::append(ost, t);
  return ost;
}

// rows : the rows of the 2-D co-ordinates
//
// columns : the columns of the 2-D co-ordinates
//
// expectedUniqueness : Are the co-ordinates all unique?
//
// expected : the expected, unique co-ordinates.
void baseTest(const Rows &rows,
              const Columns &columns,
              bool expectedUniqueness,
              const Coords &expected) {

  if (expectedUniqueness != GridPointHelper::allUnique(rows, columns)) {
    std::ostringstream oss;
    oss << "Expected unique coords ? " << expectedUniqueness << ". "
        << "This with rows=" << rows << ", and columns=" << columns;
    throw poprithms::test::error(oss.str());
  }

  auto unq = GridPointHelper::getUnique(rows, columns);
  std::sort(unq.begin(), unq.end());

  if (unq != expected) {
    std::ostringstream oss;
    oss << "Incorrect unique vector, "
        << "\nObserved=" << unq;
    oss << ", \nexpected=" << expected << ". "
        << "This with \nrows=" << rows << ",and \nColumns=" << columns;
    throw poprithms::test::error(oss.str());
  }
}

void test0() { baseTest({0, 0}, {3, 4}, true, {{0, 3}, {0, 4}}); }

void test1() { baseTest({0, 0}, {3, 3}, false, {{0, 3}}); }

void test2() {
  const Rows rows_{0, 0, 1, 1, 0, 0};
  const Columns cs{0, 1, 2, 3, 3, 2};
  baseTest(rows_, cs, true, {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}});
}

void test3() {
  const Rows rows_ = {0, 0, 1, 0, 1};
  const Columns cs = {0, 1, 2, 3, 2};
  const Coords expected{{0, 0}, {0, 1}, {0, 3}, {1, 2}};
  baseTest(rows_, cs, false, expected);
}

} // namespace

int main() {
  test0();
  test1();
  test2();
  test3();
  return 0;
}
