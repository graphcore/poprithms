// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/error/error.hpp>
#include <poprithms/memory/nest/region.hpp>

namespace {
using namespace poprithms::memory::nest;

void assertRanks(const Region &reg, uint64_t expectedRank) {
  if (reg.rank_u64() != expectedRank) {
    std::ostringstream oss;
    oss << "Expected " << reg << " to be of rank " << expectedRank << ", not "
        << reg.rank_u64();
    throw poprithms::test::error(oss.str());
  }
}
void rankTest() {
  assertRanks(Region::createFull({}), 0);
  assertRanks(Region::createFull({10}), 1);
  assertRanks({{2, 3, 4}, {{{}}, {{}}, {{}}}}, 3);
  assertRanks({{2, 0, 4, 0}, {{{}}, {{}}, {{}}, {{}}}}, 4);
}

void assertNelms(const Region &reg, int64_t expected) {
  if (reg.totalElms() != expected) {
    std::ostringstream oss;
    oss << "Expected " << reg << " to have exactly " << expected
        << " element(s), not " << reg.totalElms() << '.';
    throw poprithms::test::error(oss.str());
  }

  if ((reg.totalElms() == 0) != reg.empty()) {
    throw poprithms::test::error(
        "Error in asserNelms : empty and actual count don't agree");
  }
}

void nelmsTest() {
  assertNelms(Region::createFull({}), 1);
  assertNelms(Region::createFull({1}), 1);
  assertNelms(Region::createFull({5, 2}), 10);
  assertNelms(Region::createFull({5, 2, 0, 1}), 0);
  assertNelms(Region({}, {}), 1);
  assertNelms(Region({1}, {{{{1, 1, 1}}}}), 0);
  assertNelms(Region({2}, {{{{1, 1, 1}}}}), 1);
  assertNelms(Region({10}, {{{{4, 6, 2}}}}), 4);
  assertNelms(Region({10}, {{{{1, 2, 0}}}}), 4);
  assertNelms(Region({10}, {{{{1, 2, 1}}}}), 3);
  // combination of 2 above
  assertNelms(Region({10, 10}, {{{{1, 2, 0}}}, {{{1, 2, 1}}}}), 12);
}

} // namespace

int main() {
  rankTest();
  nelmsTest();
}
