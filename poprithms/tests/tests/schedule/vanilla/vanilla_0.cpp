// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <random>

#include <poprithms/schedule/vanilla/error.hpp>
#include <poprithms/schedule/vanilla/vanilla.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms::schedule::vanilla;

void test0() {

  for (auto eic : {ErrorIfCycle::Yes, ErrorIfCycle::No}) {
    for (auto ve : {VerifyEdges::Yes, VerifyEdges::No}) {
      const auto sched = getSchedule_i64({{1}, {2}, {}}, eic, ve);
      if (sched != std::vector<int64_t>{0, 1, 2}) {
        throw error("incorrect schedule");
      }
    }
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
    throw error("Failed to catch error when cycle");
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
    throw error("Failed to catch error when invalid edge");
  }
}

} // namespace

int main() {
  test0();
  test1();
  test2();
  return 0;
}
