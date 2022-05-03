// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <gmock/gmock.h>

#include <poprithms/autodiff/automatic/gradinfos.hpp>

using ::testing::AtLeast;
TEST(PuppeteerTest, Walking) {
  using namespace poprithms::autodiff::automatic;
  GradInfos gInfos;
  EXPECT_THROW(auto x = gInfos.grad(OpId(0), 0), poprithms::error::error)
      << "No Ops have grads set";

  EXPECT_THROW(auto x = gInfos.at(SubGraphId::createSubGraphId(0)),
               poprithms::error::error)
      << "No sub-graph 0";
}
