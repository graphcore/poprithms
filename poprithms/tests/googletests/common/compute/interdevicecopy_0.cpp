// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <cmath>
#include <gmock/gmock.h>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/op.hpp>
#include <poprithms/common/compute/ops/init.hpp>
#include <poprithms/common/compute/ops/reffrom.hpp>
#include <poprithms/common/compute/ops/viewchange.hpp>
#include <poprithms/common/compute/simexecutable.hpp>
#include <poprithms/common/compute/slickgraph.hpp>

namespace {

using namespace poprithms::common::compute;

}

TEST(CommonComputeInterDeviceCopy, Test0) {

  int64_t replicationFactor_i64{4};
  auto replicationFactor = ReplicationFactor::create(replicationFactor_i64);
  SlickGraph g(10, replicationFactor);

  auto sg0 = g.createSubGraph("sg0");

  // Shape of the tensor on the ipu:
  Shape ipuShape{3};

  int64_t circularBufferCount{3};

  // Shape of the tensor on the host:
  auto hostShape = SlickGraph::getHostShape(
      circularBufferCount, replicationFactor, ipuShape);

  EXPECT_EQ(hostShape,
            Shape({circularBufferCount, replicationFactor_i64, 3}));

  // host -> ipu -> (math) -> back to host.
  auto hostTensor       = sg0.variable(DType::Int32, hostShape, g.host());
  auto ipuTensor        = hostTensor.hostToIpu(g.rootIpu());
  ipuTensor             = ipuTensor * ipuTensor.constant(2);
  auto backToHostTensor = ipuTensor.ipuToHost(circularBufferCount);

  g.setRunnable({sg0});

  SimExecutable se(g);
  auto hostInputValue =
      HostTensor::randomInt32(0, 10, hostTensor.shape(), 1011);
  se.setHostValue(hostTensor, hostInputValue);

  se.run(sg0);
  se.getHostValue(backToHostTensor)
      .at(0)
      .assertAllEquivalent(hostInputValue.at(0).mul(2),
                           "after one iteration, just the first element of "
                           "the input has been processed.");

  // Index 1 has not yet been processed:
  auto whereDifferent =
      se.getHostValue(backToHostTensor).at(1) != hostInputValue.at(1).mul(2);
  EXPECT_EQ(whereDifferent.allZero(), false);

  // Process the remaining slices:
  for (int i = 1; i < circularBufferCount; ++i) {
    se.run(sg0);
  }

  se.getHostValue(backToHostTensor)
      .assertAllEquivalent(hostInputValue.mul(2),
                           "after the full tensor has been iterated through");

  se.run(sg0);
  se.getHostValue(backToHostTensor)
      .assertAllEquivalent(
          hostInputValue.mul(2),
          "after the full tensor has been iterated through, and an "
          "additional call (check that resets to index 0 correctly)");
}

TEST(CommonComputeInterDeviceCopy, TestBadShapes0) {

  int64_t rf{4};
  int64_t cbc{3};
  SlickGraph g(10, ReplicationFactor::create(4));

  auto sg0 = g.createSubGraph("sg0");

  // The "correct" tensors.
  auto hostTensor =
      sg0.variable(DType::Int32, Shape({cbc, rf, 17}), g.host());
  auto ipuTensor = sg0.variable(DType::Int32, Shape({17}), g.rootIpu());
  auto update0   = hostTensor.updateFromIpu_(ipuTensor);
  auto update1   = ipuTensor.updateFromHost_(hostTensor);

  // Host tensor has wrong shape.
  EXPECT_THROW(
      hostTensor.variable(Shape{cbc, rf * 2, 17}).updateFromIpu_(ipuTensor),
      poprithms::error::error);

  // Host tensor has another wrong shape.
  EXPECT_THROW(hostTensor.variable(Shape{rf, 17}).updateFromIpu_(ipuTensor),
               poprithms::error::error);

  // Host tensor has wrong type.
  EXPECT_THROW(hostTensor.variable(DType::Int64).updateFromIpu_(ipuTensor),
               poprithms::error::error);

  // Host tensor ... isn't on host!
  EXPECT_THROW(hostTensor.variable(g.rootIpu()).updateFromIpu_(ipuTensor),
               poprithms::error::error);

  // ipu tensor has wrong shape
  EXPECT_THROW(ipuTensor.variable(Shape{23}).updateFromHost_(hostTensor),
               poprithms::error::error);

  (void)update0;
  (void)update1;
}
