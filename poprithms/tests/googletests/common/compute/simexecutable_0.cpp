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

} // namespace

TEST(CommonComputeSimExecutable, BasicReduceProduct) {

  SlickGraph g;

  auto sg0 = g.createSubGraph("sg0");
  auto in0 = sg0.variable(DType::Int32, {2}, g.host());
  auto out = in0.reduceProduct(Shape({}));

  g.setRunnable({sg0});

  SimExecutable se(g);
  se.setHostValue<int>(in0, {2, 3});
  se.run(sg0);
  se.getHostValue(out).assertAllEquivalent(HostTensor::int32(6));
}

TEST(CommonComputeSimExecutable, ViewChangeOps0) {
  SlickGraph g;
  auto sg0  = g.createSubGraph("sg0");
  auto in0  = sg0.variable(DType::Int32, {2, 3}, g.host());
  auto out0 = in0.slice_({0, 1}, {2, 3}).flatten_().add_(in0.constant(1.));

  g.setRunnable({sg0});
  SimExecutable se(g);

  auto v0 = HostTensor::int32({2, 3}, {0, 1, 2, 3, 4, 5});
  se.setHostValue(in0, v0.copy());
  se.run(sg0);
  se.getHostValue(out0).assertAllEquivalent(
      v0.slice_({0, 1}, {2, 3}).flatten_().add(1));
}

TEST(CommonComputeSimExecutable, PadWithBroadcast0) {
  SlickGraph g;
  auto sg0  = g.createSubGraph("sg0");
  auto in0  = sg0.variable(DType::Int32, {2, 1}, g.host());
  auto out0 = in0.padWithBroadcastConstZero_({1, 0}, {0, 1});

  g.setRunnable({sg0});
  SimExecutable se(g);

  /**
   *  [[10]
   *   [12]]
   *
   * to
   *
   *  [[ 0 0]
   *   [10 0]
   *   [12 0]]
   *
   * */
  auto v0 = HostTensor::int32({2, 1}, {10, 12});
  se.setHostValue(in0, v0.copy());
  se.run(sg0);
  se.getHostValue(out0).assertAllEquivalent(
      HostTensor::int32({3, 2}, {0, 0, 10, 0, 12, 0}));
}

TEST(CommonComputeSimExecutable, MatMul0) {

  // The 2 tensors which we will multiply together with
  // (1) the compute::Graph (which we're testing).
  // (2) using the host tensor class (which we assume is correct).
  const auto t0 = HostTensor::uniformFloat64(-1, 1, {3, 4, 2}, 1011);
  const auto t1 = HostTensor::uniformFloat64(-1, 1, {2, 1, 2, 5}, 1011);

  // (2) the baseline:
  auto expected = t0.matmul(t1);

  // (1) Construct a computation graph with a matmul in it, construct a
  // SimExecutable and run it:
  SlickGraph g;
  auto sg0 = g.createSubGraph("sg0");

  auto in0 = sg0.variable(DType::Float64, t0.shape(), g.host());
  auto in1 = sg0.variable(DType::Float64, t1.shape(), g.host());
  auto out = in0.matmul(in1);

  g.setRunnable({sg0});
  SimExecutable se(g);

  se.setHostValue(in0, t0);
  se.setHostValue(in1, t1);
  se.run(sg0);
  auto observed = se.getHostValue(out);

  observed.assertAllClose(expected, 1e-5, 1e-5);
}

TEST(CommonComputeSimExecutable, MatMulDifferentOutType) {

  const auto t0 = HostTensor::randomInt32(-5, 5, {1, 2, 2, 4}, 1011);
  const auto t1 = HostTensor::randomInt32(-5, 5, {3, 1, 4, 3}, 1011);

  // (1) Construct a computation graph with a matmul in it, construct a
  // SimExecutable and run it:
  SlickGraph g;
  auto sg0 = g.createSubGraph("sg0");

  auto in0 = sg0.variable(DType::Int32, t0.shape(), g.host());
  auto in1 = sg0.variable(DType::Int32, t1.shape(), g.host());
  auto out = in0.matmul(in1, DType::Int64, {});

  g.setRunnable({sg0});
  SimExecutable se(g);

  se.setHostValue(in0, t0);
  se.setHostValue(in1, t1);
  se.run(sg0);
  auto observed = se.getHostValue(out);
  EXPECT_EQ(observed.dtype(), DType::Int64);
  EXPECT_EQ(observed.shape(), Shape({3, 2, 2, 3}));
}

TEST(CommonComputeSimExecutable, RemainderIsFmod0) {

  const auto t0 = HostTensor::uniformFloat64(-5, 5, {20}, 1011);
  const auto t1 = HostTensor::uniformFloat64(-1, 1, {20}, 1012);

  SlickGraph g;
  auto sg0 = g.createSubGraph("sg0");

  auto in0  = sg0.variable(DType::Float64, t0.shape(), g.host());
  auto in1  = sg0.variable(DType::Float64, t1.shape(), g.host());
  auto out0 = in0.rem(in1);
  auto out1 = in0.rem_(in1);

  g.setRunnable({sg0});
  SimExecutable se(g);

  se.setHostValue(in0, t0);
  se.setHostValue(in1, t1);

  se.run(sg0);
  auto observed0 = se.getHostValue(out0);
  auto observed1 = se.getHostValue(out1);

  auto t0_  = t0.getFloat64Vector();
  auto t1_  = t1.getFloat64Vector();
  auto out_ = observed0.getFloat64Vector();

  for (uint64_t i = 0; i < t0_.size(); ++i) {
    EXPECT_EQ(std::fmod(t0_.at(i), t1_.at(i)), out_.at(i));
  }

  observed0.assertAllEquivalent(t0.mod(t1));
  observed1.assertAllEquivalent(t0.mod(t1));
}

TEST(CommonComputeSimExecutable, EncodeOneHot0) {
  SlickGraph g;
  auto sg0     = g.createSubGraph("sg0");
  int64_t N    = 10;
  int64_t C    = 3;
  auto in0     = sg0.variable(DType::Float32, {N, C}, g.host());
  auto in1     = in0.variable();
  auto indices = sg0.variable(DType::Unsigned32, {N}, g.host());
  auto off     = in0.constant(0.25);
  auto on      = in0.constant(0.625);
  in0          = in0.encodeOneHot01_(indices);
  in1          = in1.encodeOneHotOffOn_(indices, off, on);

  g.setRunnable({sg0});
  SimExecutable se(g);

  se.setHostValue(indices, HostTensor::randomUnsigned32(0, C, {N}, 1011));

  se.run(sg0);

  auto x0 = se.getHostValue(in0);
  auto x1 = se.getHostValue(in1);

  x0.reduceSum({10, 1}).assertAllEquivalent(
      HostTensor::float32(1).expand({10, 1}), "sum of 1-hot columns of x0");

  x1.reduceSum({10, 1}).assertAllEquivalent(
      HostTensor::float32(0.625 + 0.25 + 0.25).expand({10, 1}),
      "sum of 1-hot columns of x1");

  (x0 * x1).reduceSum({10, 1}).assertAllEquivalent(
      HostTensor::float32(0.625).expand({10, 1}),
      "sum of 1-hot columns of x0*x1");
}

TEST(CommonComputeSimExecutable, DynamicUpdateMax0) {

  SlickGraph g;
  auto sg0 = g.createSubGraph("sg0");

  int64_t M{4};
  int64_t S{2};
  int64_t N{3};

  auto sliceable = sg0.hostFloat32Variable({M, S});
  auto slice     = sg0.hostFloat32Variable({N, S});
  auto offsets   = sg0.variable(DType::Unsigned32, {N}, g.host());
  auto updated   = sliceable.dynamicMultiUpdateMax_(slice, offsets);

  g.setRunnable({sg0});
  SimExecutable se(g);

  se.setHostValue<float>(sliceable, {1, 2, 3, 4, 5, 6, 7, 8});
  se.setHostValue<float>(slice, {10, 12, 11, 0, 9, 20});
  se.setHostValue<uint32_t>(offsets, {1, 2, 1});

  se.run(sg0);
  se.getHostValue(updated).assertAllEquivalent(
      HostTensor::float32({M, S}, {1, 2, 10, 20, 11, 6, 7, 8}));
}

TEST(CommonComputeSimExecutable, DynamicUpdateMaxPyTorch0) {

  // This is the example at
  // https://pytorch-scatter.readthedocs.io/en/1.3.0/functions/max.html
  //
  //
  // index   0  0  1  0  2  2  3  3
  // input   5  1  7  2  3  2  1  3
  //
  // output    5  7  3  3

  SlickGraph g;
  auto sg0       = g.createSubGraph("sg0");
  auto sliceable = sg0.hostFloat32Variable({4, 1});
  auto slice     = sg0.hostFloat32Variable({8, 1});
  auto offsets   = sg0.variable(DType::Unsigned32, {8}, g.host());
  auto updated   = sliceable.dynamicMultiUpdateMax_(slice, offsets);

  g.setRunnable({sg0});
  SimExecutable se(g);

  se.setHostValue<float>(sliceable, {0, 0, 0, 0});
  se.setHostValue<float>(slice, {5, 1, 7, 2, 3, 2, 1, 3});
  se.setHostValue<uint32_t>(offsets, {0, 0, 1, 0, 2, 2, 3, 3});

  se.run(sg0);
  se.getHostValue(updated).assertAllEquivalent(
      HostTensor::float32({4, 1}, {5, 7, 3, 3}));
}

TEST(CommonComputeSimExecutable, DynamicSlice0) {

  SlickGraph g;
  auto sg0       = g.createSubGraph("sg0");
  auto sliceable = sg0.hostFloat32Variable({2});
  int64_t nSlices{3};
  auto offset = sg0.variable(DType::Unsigned32, {nSlices, 1}, g.host());
  auto sliced = sliceable.dynamicMultiSlice(offset, Dimensions{0}, {1});

  g.setRunnable({sg0});
  SimExecutable se(g);

  se.setHostValue<uint32_t>(offset, {1, 0, 1});
  se.setHostValue<float>(sliceable, {33, 11});
  se.run(sg0);

  se.getHostValue(sliced).assertAllEquivalent(
      HostTensor::float32({3, 1}, {11, 33, 11}));
}

TEST(CommonComputeSimExecutable, DynamicSlice1) {
  SlickGraph g;
  auto sg0 = g.createSubGraph("sg0");

  auto sliceable = sg0.hostFloat32Variable({7, 2, 5});
  Dimensions dims{0, 2};
  int64_t nSlices{3};
  auto offset = sg0.variable(DType::Unsigned32, {nSlices, 2}, g.host());
  Shape sizes({4, 3});

  auto sliced = sliceable.dynamicMultiSlice(offset, dims, sizes);
  EXPECT_EQ(sliced.shape(), Shape({nSlices, 4, 2, 3}));

  g.setRunnable({sg0});
  SimExecutable se(g);

  auto vals0 = HostTensor::uniformFloat32(-1, 1, sliceable.shape(), 1011);

  // random offsets:
  auto offsets0 = HostTensor::zeros(DType::Unsigned32, {nSlices, 2});
  offsets0.dimShuffle_({{1, 0}}).at_(0).add_(
      HostTensor::randomUnsigned32(0, 2, {nSlices}, 100));
  offsets0.dimShuffle_({{1, 0}}).at_(1).add_(
      HostTensor::randomUnsigned32(0, 3, {nSlices}, 101));

  se.setHostValue(offset, offsets0);
  se.setHostValue(sliceable, vals0);

  se.run(sg0);

  auto l = offsets0.at(1).getUnsigned64Vector();
  auto u = offsets0.at(1)
               .toUnsigned64()
               .add(HostTensor::unsigned64({2}, sizes.get_u64()))
               .getUnsigned64Vector();

  vals0.slice(dims, l, u).assertAllEquivalent(se.getHostValue(sliced).at(1));
}

TEST(CommonComputeSimExecutable, DynamicUpdate0) {

  SlickGraph g;
  auto sg0       = g.createSubGraph("sg0");
  auto sliceable = sg0.hostFloat32Variable({4});
  int64_t nSlices{3};
  auto offset = sg0.variable(DType::Unsigned32, {nSlices, 1}, g.host());
  auto slice  = sliceable.variable({nSlices, 1});

  sliceable.dynamicMultiUpdate_(slice, offset, Dimensions{0});

  g.setRunnable({sg0});
  SimExecutable se(g);

  se.setHostValue<uint32_t>(offset, {1, 0, 3});
  se.setHostValue<float>(slice, {20, 30, 40});
  se.setHostValue<float>(sliceable, {10, 10, 10, 10});
  se.run(sg0);

  se.getHostValue(sliceable).assertAllEquivalent(
      HostTensor::float32({4}, {30, 20, 10, 40}));
}

TEST(CommonComputeSimExecutable, DynamicUpdate1) {

  SlickGraph g;
  auto sg0       = g.createSubGraph("sg0");
  auto sliceable = sg0.hostFloat32Variable({2, 3});
  int64_t nSlices{2};
  auto offset = sg0.variable(DType::Unsigned32, {nSlices, 2}, g.host());
  auto slice  = sliceable.variable({nSlices, 1, 2});

  sliceable.dynamicMultiUpdate_(slice, offset, Dimensions{0, 1});

  g.setRunnable({sg0});
  SimExecutable se(g);

  se.setHostValue<uint32_t>(offset, {0, 1, 1, 0});

  //  [[20 30]
  //   [40 50]]
  se.setHostValue<float>(slice, {20, 30, 40, 50});

  //  [[10  10  10]
  //   [10  10  10]]
  //
  // where the slices go: slice 0 at (0,1) and slice 1 at (1,0):
  // [[. 0 0]
  //  [1 1 .]]
  se.setHostValue<float>(sliceable, {10, 10, 10, 10, 10, 10});
  se.run(sg0);

  se.getHostValue(sliceable).assertAllEquivalent(
      HostTensor::float32({2, 3}, {10, 20, 30, 40, 50, 10}));
}

TEST(CommonComputeSimExecutable, UnfoldNumerics0) {

  SlickGraph g;
  auto sg0 = g.createSubGraph("sg0");

  auto x0 = sg0.hostInt32Variable({3, 4});
  auto y  = x0.unfold_(Dimension(1), /*size=*/1, /*step=*/2);
  auto y0 = x0.unfold_(Dimension(1), /*size=*/2, /*step=*/1);

  auto us0 = x0.slice_({0, 0}, {2, 1}).upsample_(2, Dimension(1));
  auto us1 = x0.slice_({0, 0}, {1, 2}).upsample_(2, Dimension(0));

  g.setRunnable({sg0});
  SimExecutable se(g);

  // 0  1  2  3
  // 4  5  6  7
  // 8  9 10  11
  se.setHostValue(x0, HostTensor::arangeInt32(0, 12, 1).reshape({3, 4}));
  se.run(sg0);

  // 0 2
  // 4 6
  // 8 10
  se.getHostValue(y).assertAllEquivalent(
      HostTensor::int32({3, 2, 1}, {0, 2, 4, 6, 8, 10}));

  // 0  1  1  2  2  3
  // 4  5  5  6  6  7
  // 8  9  9  10 10 11
  se.getHostValue(y0).assertAllEquivalent(
      HostTensor::int32(
          {3, 3, 2},
          {0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11}),
      "y0");

  // 0 0
  // 4 4
  se.getHostValue(us0).assertAllEquivalent(
      HostTensor::int32({2, 2}, {0, 0, 4, 4}));

  // 0 1
  // 0 1
  se.getHostValue(us1).assertAllEquivalent(
      HostTensor::int32({2, 2}, {0, 1, 0, 1}));
}
