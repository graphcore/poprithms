// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <vector>

#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/sett.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms::memory::nest;
void assertEqual(const std::vector<int64_t> &computed,
                 const std::vector<int64_t> &expected,
                 int c) {
  if (computed != expected) {
    std::ostringstream oss;
    oss << "Failed in assertEqual, case " << c << '\n';
    oss << "Computed=";
    poprithms::util::append(oss, computed);
    oss << " expected=";
    poprithms::util::append(oss, expected);
    throw error(oss.str());
  }
}
} // namespace

int main() {

  using namespace poprithms::memory::nest;

  auto computed = poprithms::memory::nest::Sett{{{{1, 0, 0}}}}.getOns(0, 5);
  assertEqual(computed, {0, 1, 2, 3, 4}, 0);

  // 11..11..11..
  std::vector<int64_t> expected{0, 1, 4, 5, 8, 9};
  computed = poprithms::memory::nest::Sett{{{{2, 2, 0}}}}.getOns(0, 12);
  assertEqual(computed, expected, 0);

  expected = {1, 4, 5, 8, 9};
  computed = poprithms::memory::nest::Sett{{{{2, 2, 0}}}}.getOns(1, 12);
  assertEqual(computed, expected, 6);

  expected = {0, 1, 4, 5, 8, 9, 12};
  computed = poprithms::memory::nest::Sett{{{{2, 2, 0}}}}.getOns(0, 13);
  assertEqual(computed, expected, 7);

  // ...11111111111.....11111111111.....11111111111.....
  //    1...1111...
  //        .1.1
  //    -    - -        -   - -         -   - -
  //    3    8 10       19  2426        35  4042
  //
  expected = {3, 8, 10, 19, 24, 26, 35, 40, 42};
  computed =
      poprithms::memory::nest::Sett{{{{11, 5, 3}, {4, 3, 4}, {1, 1, 1}}}}
          .getOns(0, 45);
  assertEqual(computed, expected, 8);

  // ....1111111111.........
  // ..1111111111111..111
  const auto sett0 =
      poprithms::memory::nest::Sett{{{{10, 100, 4}, {13, 2, -2}}}};

  computed = sett0.getOns(0, 20);
  expected = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  assertEqual(computed, expected, 12);
}
