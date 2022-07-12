// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <poprithms/common/compute/testutil/misctraintester.hpp>

int main() {
  using namespace poprithms::common::compute::testutil;
  std::make_unique<SimTester<MiscTrainTester>>()->all();
}
