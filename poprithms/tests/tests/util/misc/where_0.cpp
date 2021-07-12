// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <chrono>
#include <iostream>
#include <ostream>
#include <sstream>
#include <vector>

#include <poprithms/util/error.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/where.hpp>

namespace {

using namespace poprithms;

template <typename T>
std::ostream &operator<<(std::ostream &ost, const std::vector<T> &t) {
  util::append(ost, t);
  return ost;
}

void testWhereKeysInVals(const std::vector<int> &keys,
                         const std::vector<int> &vals,
                         const std::vector<bool> &expected) {

  auto mask = util::whereKeysInVals(keys, vals);
  if (mask != expected) {
    std::ostringstream oss;
    oss << "Failed in test of whereKeysInVals. Keys = " << keys
        << ". Vals = " << vals << " Expected = " << expected
        << " Observed = " << mask;
    throw util::error(oss.str());
  }
}
void test0() {

  testWhereKeysInVals({}, {10, 5, 2}, {});
  testWhereKeysInVals({10, 11, 10}, {}, {0, 0, 0});
  testWhereKeysInVals({10, 11, 10}, {100}, {0, 0, 0});
  testWhereKeysInVals({10, 11, 10}, {0}, {0, 0, 0});
  testWhereKeysInVals({10, 11, 10}, {11, 11}, {0, 1, 0});
  testWhereKeysInVals({10, 11, 10}, {5, 10}, {1, 0, 1});
  testWhereKeysInVals({3, 2, 1}, {1, 2, 3}, {1, 1, 1});
  testWhereKeysInVals({1, 2, 3}, {0, 4, 5}, {0, 0, 0});
  testWhereKeysInVals({1, 2, 3}, {0, 2, 5, 10, 11}, {0, 1, 0});
  testWhereKeysInVals({1, 2, 100, 3}, {100, 0, 2, 5, 10, 11}, {0, 1, 1, 0});

  std::vector<int> keys{5, 2, 7, 5, 1, 1, 10};
  std::vector<int> vals{0, 0, 1};
  testWhereKeysInVals(keys, vals, {0, 0, 0, 0, 1, 1, 0});
}

} // namespace

int main() {
  test0();

  return 0;
}
