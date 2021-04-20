// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <string>

#include <poprithms/schedule/shift/error.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>
#include <poprithms/util/printiter.hpp>

namespace {
using namespace poprithms::schedule::shift;

/********** Logging helpers ***********/

// Example expansion: LOCATION_ --> `testFunction` (line 34)
#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)
#define LOCATION_                                                            \
  ("`" + std::string(__func__) + "` (line " STRINGIFY(__LINE__) ")")

#define scheduleMismatchTestFailureStr(actual, expected)                     \
  (scheduleMismatchTestFailureStr_)(LOCATION_, (actual), (expected))

std::string
scheduleMismatchTestFailureStr_(const std::string &testFailureLocation,
                                const std::vector<OpAddress> &actual,
                                const std::vector<OpAddress> &expected) {
  using poprithms::util::append;

  // clang-format off

  std::ostringstream oss;
  oss << testFailureLocation << ": FAILED - schedule mismatch:" << std::endl
      << "    actual   = "; append(oss, actual);
  oss << std::endl
      << "    expected = "; append(oss, expected);

  // clang-format on

  return oss.str();
}

/********** Tests ***********/

enum class ThrowingTestResult { DidNotThrow = 0, DidThrow };

ThrowingTestResult testGetSubScheduleOnInvalidOpAddress() {
  Graph g;
  const auto op0 = g.insertOp("Op0");
  const auto op1 = g.insertOp("Op1");
  const auto op2 = g.insertOp("Op2");
  const auto op3 = g.insertOp("Op3");
  ScheduledGraph sg(std::move(g),
                    KahnTieBreaker::RANDOM,
                    TransitiveClosureOptimizations::allOff(),
                    RotationTermination::preStart());

  try {
    sg.getSubSchedule({op0, op3 + 1u, op1, op3 + 2u, op2, op3, op3 + 2u});
  } catch (const poprithms::error::error &e) {
    return ThrowingTestResult::DidThrow;
  }

  return ThrowingTestResult::DidNotThrow;
}

ThrowingTestResult testGetSubScheduleOnDuplicateOpAddresses() {
  Graph g;
  const auto op0 = g.insertOp("Op0");
  ScheduledGraph sg(std::move(g),
                    KahnTieBreaker::RANDOM,
                    TransitiveClosureOptimizations::allOff(),
                    RotationTermination::preStart());

  try {
    sg.getSubSchedule({op0, op0});
  } catch (const poprithms::error::error &e) {
    return ThrowingTestResult::DidThrow;
  }

  return ThrowingTestResult::DidNotThrow;
}

void testGetSubScheduleCanHandleUnsortedSubset() {
  // Setup graph: Op0 -> Op1 -> Op2.
  Graph g;
  const std::vector<OpAddress> ops = g.insertOps({"Op1", "Op2", "Op3"});
  g.insertConstraints({{ops[0], ops[1]}, {ops[1], ops[2]}});
  ScheduledGraph sg(std::move(g),
                    KahnTieBreaker::RANDOM,
                    TransitiveClosureOptimizations::allOff(),
                    RotationTermination::preStart());

  // We will test on ths subset.
  const std::vector<OpAddress> subset = {ops[2], ops[1]};

  // The expected schedule is the OpAddresses in the subset, ordered according
  // to the above topology.
  const std::vector<OpAddress> expected = {ops[1], ops[2]};

  // Do test.
  const auto actual = sg.getSubSchedule(subset);

  if (actual != expected) {
    throw error(scheduleMismatchTestFailureStr(actual, expected));
  }
}

void testGetSubScheduleOnUserOpsOnlyDoesNotContainInternalOps() {
  // Setup graph:
  //
  // Op0 -----|
  //  |       |
  //  |       V
  //  |      Bin ----> Op2
  //  |       ^
  //  V       |
  // Op1 -----|
  //
  // Giving linearised sub-schedule on `ops`: Op0 -> Op1 -> Op2.
  Graph g;
  const std::vector<OpAddress> ops = g.insertOps({"Op1", "Op2", "Op3"});
  g.insertBinConstraints({{ops[0], ops[1]}, {ops[2]}}, "bin-");
  g.insertConstraint(ops[0], ops[1]);
  ScheduledGraph sg(std::move(g));

  // `ops` only contains the three manually inserted Ops, not the bin Op, so
  // expected will only contain those three.
  const std::vector<OpAddress> expected = ops;

  // Do test.
  const auto actual = sg.getSubSchedule(ops);

  if (actual != expected) {
    throw error(scheduleMismatchTestFailureStr(actual, expected));
  }
}

void testGetSubScheduleSameAsViewInternalWhenNoInternalOps() {
  Graph g;
  const auto op0   = g.insertOp("Op0");
  const auto op1   = g.insertOp("Op1");
  const auto op2   = g.insertOp("Op2");
  const auto alloc = g.insertAlloc(2.0);

  g.insertOpAlloc({op1, op2}, alloc);
  g.insertConstraint(op1, op2);

  ScheduledGraph sg(std::move(g),
                    KahnTieBreaker::RANDOM,
                    TransitiveClosureOptimizations::allOff(),
                    RotationTermination::preStart());

  const auto actual    = sg.getSubSchedule({op0, op1, op2});
  const auto &expected = sg.viewInternalScheduleToOp();

  if (actual != expected) {
    throw error(scheduleMismatchTestFailureStr(actual, expected));
  }
}

} // namespace

int main() {

  if (testGetSubScheduleOnInvalidOpAddress() !=
      ThrowingTestResult::DidThrow) {
    throw error("Calling getSubSchedule on non-existant Op did not throw");
  }

  if (testGetSubScheduleOnDuplicateOpAddresses() !=
      ThrowingTestResult::DidThrow) {
    throw error("Calling getSubSchedule on duplicate ops did not throw");
  }

  testGetSubScheduleCanHandleUnsortedSubset();

  testGetSubScheduleOnUserOpsOnlyDoesNotContainInternalOps();

  testGetSubScheduleSameAsViewInternalWhenNoInternalOps();

  return 0;
}
