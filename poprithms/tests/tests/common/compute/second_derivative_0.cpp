// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <iostream>

#include <poprithms/common/compute/autodiff/autodiffer.hpp>
#include <poprithms/common/compute/simexecutable.hpp>
#include <poprithms/common/compute/slickgraph.hpp>

int main() {

  using namespace poprithms::common::compute;

  SlickGraph graph;
  auto sg0 = graph.createSubGraph("sg0");

  int64_t nElms = 7;
  auto x0       = sg0.hostFloat32Variable({nElms});
  auto loss     = x0.mul(x0).mul(x0.reverse(0)).reduceSum();

  Autodiffer ad(graph);

  // Graph to compute second derivative of loss w.r.t. x0:
  auto hc      = ad.completeHessian(loss, x0);
  auto sg2     = graph.subGraph(hc.hessianGraph);
  auto x2      = graph.tensor(hc.targetInHessianGraph);
  auto hessian = graph.tensor(hc.hessian);

  // Run the graph:
  graph.setRunnable({sg2});
  SimExecutable cms(graph);
  auto init0 = HostTensor::randomInt32(1, 6, x0.shape(), 1011).toFloat32();
  cms.setHostValue(x2, init0);
  cms.run(sg2);

  auto observed = cms.getHostValue(hessian);

  // second derivative computation:
  //
  // loss   =   x[0]^2 * x[4] +
  //            x[1]^2 * x[3] +
  //            x[2]^2 * x[2] +
  //            x[3]^2 * x[1] +
  //            x[4]^2 * x[0].
  //
  // dLoss =  [ 2 * x[0] * x[4] + x[4]^2
  //            2 * x[1] * x[3] + x[3]^2
  //            2 * x[2] * x[2] + x[2]^2
  //            2 * x[3] * x[1] + x[1]^2
  //            2 * x[4] * x[0] + x[0]^2 ]
  //
  // ddLoss[0,0] = 2 * x[4]
  // ddLoss[0,1] = 0
  // ...
  // etc.
  auto expected = HostTensor::zeros(DType::Float32, observed.shape());
  for (int64_t i = 0; i < nElms; ++i) {
    for (int64_t j = 0; j < nElms; ++j) {
      float v = 0;
      if (i == j && i + j == nElms - 1) {
        v = 6 * (init0.getFloat32(i));
      } else if (i == j) {
        v = 2 * (init0.getFloat32(nElms - i - 1));
      } else if (i + j == nElms - 1) {
        v = 2 * ((init0.getFloat32(i) + init0.getFloat32(nElms - i - 1)));
      }
      expected.at_(i).at_(j).update_(HostTensor::float32(v));
    }
  }

  //  [[ 4  0  0 0  0 0  12 ]
  //   [ 0  2  0 0  0 10 0  ]
  //   [ 0  0  2 0  4 0  0  ]
  //   [ 0  0  0 24 0 0  0  ]
  //   [ 0  0  4 0  2 0  0  ]
  //   [ 0  10 0 0  0 8  0  ]
  //   [ 12 0  0 0  0 0  8  ]]
  std::cout << observed.toInt16() << std::endl;
  expected.assertAllEquivalent(observed);
}
