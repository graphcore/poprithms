// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_TESTUTIL_MISCTRAINTESTER_HPP
#define POPRITHMS_COMMON_COMPUTE_TESTUTIL_MISCTRAINTESTER_HPP

#include <poprithms/common/compute/testutil/polyexecutabletester.hpp>

namespace poprithms {
namespace common {
namespace compute {
namespace testutil {

/**
 * Miscellaneous training examples.
 * */
class MiscTrainTester : public PolyExecutableTester {
public:
  MiscTrainTester();
  virtual ~MiscTrainTester();

  void testRepeatedAddInput();

  void all() { testRepeatedAddInput(); }
};

} // namespace testutil
} // namespace compute
} // namespace common
} // namespace poprithms

#endif
