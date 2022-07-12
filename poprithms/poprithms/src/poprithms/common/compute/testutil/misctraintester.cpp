// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include <poprithms/common/compute/autodiff/autodiffer.hpp>
#include <poprithms/common/compute/autodiff/automaticquerier.hpp>
#include <poprithms/common/compute/prune/pruner.hpp>
#include <poprithms/common/compute/slickgraph.hpp>
#include <poprithms/common/compute/testutil/finitedifference.hpp>
#include <poprithms/common/compute/testutil/misctraintester.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/program/prune/prune.hpp>

namespace poprithms {
namespace common {
namespace compute {
namespace testutil {

using Autodiffer = Autodiffer<SlickGraph, AutomaticMutator>;

void MiscTrainTester::testRepeatedAddInput() {

  // The 'ir'.
  SlickGraph m(1000, ReplicationFactor::create(1));

  //  The callee graph.
  SubGraph sg0 = m.createSubGraph("sg0");
  Tensor x     = sg0.hostFloat64Variable({3, 4});
  Tensor w     = x.variable({4, 5});
  Tensor b     = x.variable(Shape{5});
  Tensor add0  = x.matmul(w) + b;
  Tensor add2  = add0 + add0;

  // The main graph, which will call into the callee.
  auto sg2     = m.createSubGraph("sg2");
  Tensor xMain = x.variable(sg2);
  localAssert(
      xMain.shape() == x.shape() && xMain.dtype() == x.dtype() &&
          xMain.subGraphId() == sg2,
      "The tensor.variable(foo) methods create new variables which are like "
      "tensor is all ways excepting 'foo'. In this case we're creating a "
      "variable like x but in a different subgraph.");

  Tensor wMain = w.variable(sg2);
  Tensor bMain = b.variable(sg2);

  // call, with all required copies.
  // All tensors in the callee are copied out: in late pruning we trust.
  auto c0 = sg2.callAllOut(sg0, {{{xMain, x}, {wMain, w}, {bMain, b}}});

  Tensor loss = (add0.dstInCaller(c0) * add2.dstInCaller(c0)).reduceSum();
  localAssert(loss.subGraphId() == sg2, "loss is in the main scope");

  // Do autodiff in-graph, the lazy well (creates grad graph for sg0, extends
  // main graph with gradients).
  auto dW = Autodiffer(m).backward(loss, {wMain})[0];

  (void)dW;

  // specify which subgraphs the user can run (ala poplar::Engine).
  m.setRunnable({sg2});

  setCompiledSlickGraph(m);

  // Run a finite difference test of numerical correctness of gradients.
  // This is an "integration" test, but it's very fast as there's no poplar.
  std::unordered_map<TensorId, HostTensor> initVals;
  initVals.insert({wMain, HostTensor::uniformFloat64(-1., 1, {4, 5}, 312)});
  initVals.insert({xMain, HostTensor::uniformFloat64(-1., 1, {3, 4}, 313)});
  initVals.insert({bMain, HostTensor::uniformFloat64(-1., 1, {5}, 314)});
  poprithms::common::compute::testutil::finiteDifferenceTest(
      cm(),
      loss,
      wMain,
      m.tensor(dW),
      initVals,
      /** seed for perturbation */ 1011,
      /** perturbation size: */ 1e-10);
}

} // namespace testutil
} // namespace compute
} // namespace common
} // namespace poprithms
