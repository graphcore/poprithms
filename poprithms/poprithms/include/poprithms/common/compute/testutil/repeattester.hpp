// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_TESTUTIL_REPEATTESTER_HPP
#define POPRITHMS_COMMON_COMPUTE_TESTUTIL_REPEATTESTER_HPP

#include <poprithms/common/compute/testutil/polyexecutabletester.hpp>

namespace poprithms {
namespace common {
namespace compute {
namespace testutil {

/**
 * An abstract class for running tests of the Repeat op with require an
 * executable (numerical tests). Classes which inherit from this class will
 * implement #getCompiledSlickGraph method.
 * */
class RepeatTester : public PolyExecutableTester {
public:
  RepeatTester()          = default;
  virtual ~RepeatTester() = default;
  void testRepeatCarriedPreStackedToLoss();
  void testRepeatStackedToLossPreCarried();
  void testDynamicUpdateInRepeat0();
  void testOffPathCarriedIncr();
  void testShardedSinTrain0();
  void testMultiProngOut0();
  void testLastMinuteZero();
  void testTowardsNoExit0();
  void testRepeatInCall0();
  void testStackedInput0();
  void testReverseOrder0();
  void testIsStackedOut();
  void testSimpleInfer0();
  void testMultiInCall0();
  void testGlobalPower0();
  void testCrissCross0();
  void testMixedBag0();
  void testAutodiff0();
  void testVisited0();
  void testProduct0();
  void testLadder0();
  void testRepeat0();
  void testManual0();

  // All of the tests listed above:
  void all() {
    testRepeatCarriedPreStackedToLoss();
    testRepeatStackedToLossPreCarried();
    testDynamicUpdateInRepeat0();
    testOffPathCarriedIncr();
    testShardedSinTrain0();
    testMultiProngOut0();
    testLastMinuteZero();
    testTowardsNoExit0();
    testRepeatInCall0();
    testStackedInput0();
    testReverseOrder0();
    testIsStackedOut();
    testSimpleInfer0();
    testMultiInCall0();
    testGlobalPower0();
    testCrissCross0();
    testMixedBag0();
    testAutodiff0();
    testVisited0();
    testProduct0();
    testLadder0();
    testRepeat0();
    testManual0();
  }
};

} // namespace testutil
} // namespace compute
} // namespace common
} // namespace poprithms

#endif
