// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <random>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/vanilla/vanilla.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms::schedule::vanilla;

void test0() {

  for (auto eic : {ErrorIfCycle::Yes, ErrorIfCycle::No}) {
    for (auto ve : {VerifyEdges::Yes, VerifyEdges::No}) {
      const auto sched = getSchedule_i64({{1}, {2}, {}}, eic, ve);
      if (sched != std::vector<int64_t>{0, 1, 2}) {
        throw poprithms::test::error("incorrect schedule");
      }
    }
  }
}

void testRepeatedEdge() {
  // 0 -> {}
  // 1 -> {2,2,3,2}  <-- repeated edge 1->2
  // 2 -> {3}
  // 3 -> {0}
  const auto sched = getSchedule_u64(
      {{}, {2, 2, 3, 2}, {3}, {0}}, ErrorIfCycle::Yes, VerifyEdges::Yes);
  if (sched != std::vector<uint64_t>{1, 2, 3, 0}) {
    throw poprithms::test::error("Failed test with repeated edge, 1->2");
  }
}

void test1() {

  bool caught{false};
  try {
    const auto sched =
        getSchedule_u64({{1}, {0}}, ErrorIfCycle::Yes, VerifyEdges::Yes);
  } catch (const poprithms::error::error &) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error("Failed to catch error when cycle");
  }
}

void test2() {

  bool caught{false};
  try {
    const auto sched =
        getSchedule_u64({{2}, {0}}, ErrorIfCycle::Yes, VerifyEdges::Yes);
  } catch (const poprithms::error::error &) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error("Failed to catch error when invalid edge");
  }
}

void testUnique0() {
  auto uni = Query<int64_t>::hasUniqueSchedule({{1}, {3}, {}, {2}},
                                               VerifyEdges::Yes);
  if (uni == false) {
    throw poprithms::test::error("There is a unique schedule, 0,1,3,2");
  }
}

void testUnique1() {
  auto uni = Query<int64_t>::hasUniqueSchedule({{1}, {2, 3}, {4}, {4}, {}},
                                               VerifyEdges::Yes);
  if (uni == true) {
    throw poprithms::test::error(
        "There is not a unique schedule, either 2 or 3 may appear third");
  }
}

void testUnique2() {
  auto uni = Query<uint64_t>::hasUniqueSchedule({{1}, {3}, {1}, {2}},
                                                VerifyEdges::Yes);
  if (uni == true) {
    throw poprithms::test::error(
        "There is not a unique schedule, as there is a cycle");
  }
}

} // namespace

int main() {
  test0();
  test1();
  test2();
  testUnique0();
  testUnique1();
  testUnique2();
  testRepeatedEdge();
  return 0;
}
