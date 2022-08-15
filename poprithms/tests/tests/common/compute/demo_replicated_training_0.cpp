// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <iostream>

#include <poprithms/common/compute/autodiff/autodiffer.hpp>
#include <poprithms/common/compute/callstackquerier.hpp>
#include <poprithms/common/compute/ops/reduce.hpp>
#include <poprithms/common/compute/simexecutable.hpp>
#include <poprithms/common/compute/slickgraph.hpp>

namespace {

using poprithms::program::callstack::StackTensorId;
using poprithms::program::callstack::StackTensorIds;
using namespace poprithms::common::compute;

/**
 * In this example, there are 3 graphs/programs which the user can run:
 *
 * 1) for copying model weights to ipu (from host)
 * 2) for copying model weights to host (from ipu)
 * 3) for copying data to ipu and performing 1 training step:
 *    - Stream data to ipu.
 *    - Perform forward pass, compute loss.
 *    - Stream loss back to host.
 *    - Perform backward pass.
 *    - Perform weight update.
 *
 * The user can call 1, 2, 3 at any time from their application.
 * */

void dataReplication0() {

  /////////////////////////////////////////////////////
  /// Construct the graphs, describe the computation //
  /////////////////////////////////////////////////////
  int64_t replicationFactor{4};
  uint64_t tilesPerReplica{32};
  int64_t nIterations{100};

  SlickGraph ir(tilesPerReplica,
                ReplicationFactor::create(replicationFactor));

  // fwd
  auto sgFwdBwdWu = ir.createSubGraph("sgFwdBwdWu");
  auto w0         = sgFwdBwdWu.rootIpuFloat32Variable({2, 2});
  auto d0         = w0.variable();
  auto loss = (w0.matmul(d0 - d0) + w0).reduceSum().reduceSumAcrossReplicas();

  // bwd
  auto dw0id = Autodiffer(ir).backward(loss, {w0})[0];
  auto dw0   = ir.tensor(dw0id);

  // wu
  float learningRate{0.01};
  auto lr = w0.constant(learningRate);
  w0      = w0.sub_(dw0.mul_(lr));

  // training graph
  auto sgTrain = ir.createSubGraph("sgTrainStep");
  const Shape hostDataShape{nIterations, replicationFactor, 2, 2};
  auto hostData   = sgTrain.hostFloat32Variable(hostDataShape);
  auto d1         = hostData.hostToIpu(ir.rootIpu());
  auto c0         = sgTrain.call(sgFwdBwdWu, {{{d1, d0}}}, {loss});
  auto hostLosses = loss.dstInCaller(c0).ipuToHost(nIterations);

  // weights from host to ipu.
  auto sgWeightsToIpu = ir.createSubGraph("hostToIpu_weights");
  auto wHost          = sgWeightsToIpu.hostFloat32Variable({1, 1, 2, 2});
  w0.refTo_(sgWeightsToIpu).updateFromHost_(wHost);

  // weights from ipu to host.
  auto sgWeightsToHost = ir.createSubGraph("ipuToHost_weights");
  auto finalHost       = w0.refTo_(sgWeightsToHost).ipuToHost(1);

  // User will call these 3 sub-graphs directly, at runtime.
  ir.setRunnable({sgTrain, sgWeightsToIpu, sgWeightsToHost});

  ////////////////////////
  /// Compile the graph //
  ////////////////////////
  SimExecutable compiledMachine(ir);

  ////////////////////////////////
  /// Run the compiled programs //
  ////////////////////////////////
  // set weights from host.
  auto wHost0 = HostTensor::uniformFloat32(-1, 1, {1, 1, 2, 2}, 1012);
  compiledMachine.setHostValue(wHost, wHost0);
  compiledMachine.run(sgWeightsToIpu);

  // set data from host.
  compiledMachine.setHostValue(
      hostData, HostTensor::uniformFloat32(-1, 1, hostDataShape, 1011));

  // train for multiple interations.
  for (int64_t i = 0; i < nIterations; ++i) {
    compiledMachine.run(sgTrain);
  }

  // get the trained weights.
  compiledMachine.run(sgWeightsToHost);

  /////////////////////////////////////////
  /// Perform numerical tests on results //
  /////////////////////////////////////////
  auto trainedWeights = compiledMachine.getHostValue(finalHost);

  // Use algebra + calculus.
  //
  // Let,
  //   rf                    = replicationFactor
  //
  // Then,
  //   Loss                  =  rf * weights.reduceSum()
  //   dLoss/dWeight_{ij}    =  rf
  //
  // So,
  //   weights_{t+1}         = weights_{t} - lr * rf.
  //   loss_{t+1} - loss_{t} = rf * rf * lr.

  (wHost0 - compiledMachine.getHostValue(finalHost))
      .assertAllClose(
          HostTensor::float32(replicationFactor * learningRate * nIterations),
          1e-3,
          1e-3);

  auto losses = compiledMachine.getHostValue(hostLosses)
                    .slice({0, 0}, {nIterations, 1})
                    .squeeze();

  (losses.slice({0}, {nIterations - 1}) - losses.slice({1}, {nIterations}))
      .assertAllClose(HostTensor::float32(2 * 2 * replicationFactor *
                                          replicationFactor * learningRate)
                          .expand({nIterations - 1}),
                      1e-3,
                      1e-3);
}

// Transform to replace replica reductions whose input is the same across all
// replicas with a multiply-by-replication-factor.
void removeRedundantReplicaReductions(SubGraph sg0) {

  auto &g = sg0.graph();
  CallstackQuerier q(g);
  StackTensorIds traversalStarts;
  auto redOps = g.opIds<ReduceSumAcrossReplicas>(sg0);

  const auto rf = g.replicationFactor_u64();

  // Starting at all the inputs to the graph (we conservatively assume they
  // all have different values on replicas).
  for (auto x : sg0.varInitIds()) {
    traversalStarts.push_back(StackTensorId({x, 0}, {}));
  }

  // traverse forward from the starting points, halting at replica
  // reductions. Store all tensors visited in this set:
  std::set<OpId> multiReplicaVals;
  for (auto x : q.onMultiGraphPathFrom(traversalStarts, [&](auto x) {
         return g.dynamicCast<ReduceSumAcrossReplicas>(x.tId().opId()) ==
                nullptr;
       })) {
    multiReplicaVals.insert(x.tId().opId());
  }

  // For each of the replica reduction ops whose input is not in the
  // traversed set, replace with a scale-by-replication factor.
  for (auto redOp : redOps) {
    auto redIn = g.inTensorId(redOp, 0);
    if (multiReplicaVals.count(redIn.opId()) == 0) {
      auto scaled = Tensor(redIn, &g).mul(rf);
      g.removeOp(
          redOp,
          OptionalTensorIds{scaled.id()},
          "Replacing replica-sum-reduce with scale-by-replication-"
          "factor, as the values on all replicas of the input are equal.");
    }
  }
}

void tensorParallel0() {

  //
  // MLP tensor parallel as per Megatron paper Section 3:
  //  "Hence, we partition the first GEMM in this column parallel
  //   fashion and split the second GEMM along its rows."
  //
  // Megatron paper: https://arxiv.org/pdf/1909.08053.pdf
  //
  //
  // All data and weight is NxN (N = 2).
  //
  // First mlp weights, split by columns.
  //
  // w0 : [[ 1 2 ]
  //       [ 3 4 ]]
  //
  //  ==>
  //
  // w0_0 : [[ 1 ]
  //         [ 3 ]]
  //
  // w0_1 : [[ 2 ]
  //         [ 4 ]]

  int64_t N{2};
  auto w0 = HostTensor::uniformFloat32(-1, 1, {N, N}, /* seed = */ 1011);

  // Second mlp weights. Split by rows.
  //
  // w1 : [[ 5 6 ]
  //       [ 7 8 ]]
  //
  // ==>
  //
  // w1_0 : [[ 5 6 ]]
  //
  // w1_1 : [[ 7 8 ]]
  //
  auto w1 = HostTensor::uniformFloat32(-1, 1, {N, N}, /* seed = */ 1012);

  ////////////////
  /// Build IR ///
  ////////////////
  auto mlp = [](Tensor data, Tensor w0, Tensor w1) {
    return data.matmul(w0)
        .abs()
        .sqrt() // This is "gelu" in the megatron paper.
        .matmul(w1)
        .reduceSumAcrossReplicas() // This is the "g" in the megatron paper.
        .abs()
        .sqrt(); // This is "dropout" in the megatron paper.
  };

  // Identical to #mlp, but without the cross-replica reduction.
  auto hostMlp = [](Tensor data, Tensor w0, Tensor w1) {
    return data.matmul(w0).abs().sqrt().matmul(w1).abs().sqrt();
  };

  SlickGraph g(/* nTiles = */ 32, ReplicationFactor::create(N));

  // Baseline (b_). Just do the computation on host without any splitting.
  auto sgBaseline = g.createSubGraph("sgBaseline");

  // 3 tensors of shape NxN.
  auto b_w0    = sgBaseline.hostFloat32Variable({N, N});
  auto b_w1    = b_w0.variable();
  auto b_data  = b_w0.variable();
  auto b_loss  = hostMlp(b_data, b_w0, b_w1).reduceSum();
  auto b_grads = Autodiffer(g).backward(b_loss, {b_w0, b_w1});

  // Tensor parallel version.
  auto sgTensorParallel = g.createSubGraph("sgTensorParallel");

  // Host tensors, all NxN.
  auto host_w0   = sgTensorParallel.hostFloat32Variable({N, N});
  auto host_w1   = host_w0.variable();
  auto host_data = host_w0.variable();

  // broadcast data to all replicas.
  auto ipu_data = host_data.reshape_({1, 1, N, N}).hostToIpu(g.rootIpu());

  // As w0 is split by columns, we need some transposes to get the correct
  // slices onto the correct replicas.
  auto ipu_w0 = host_w0.dimShuffle({{1, 0}})
                    .reshape_({1, N, 1, N})
                    .hostToIpu(g.rootIpu())
                    .dimShuffle({{1, 0}});

  auto ipu_w1 = host_w1.reshape({1, N, 1, N}).hostToIpu(g.rootIpu());

  auto loss = mlp(ipu_data, ipu_w0, ipu_w1).reduceSum(Shape{}).div(N);

  auto tParallelGrads = Autodiffer(g).backward(loss, {host_w0, host_w1});

  g.setRunnable({sgTensorParallel, sgBaseline});

  auto compileRunVerify = [&]() {
    //////////////////////
    /// Compile and run //
    //////////////////////
    SimExecutable se(g);
    se.setHostValue(host_w0, w0);
    se.setHostValue(host_w1, w1);

    se.setHostValue(b_w0, w0);
    se.setHostValue(b_w1, w1);

    se.run(sgTensorParallel);
    se.run(sgBaseline);

    ////////////////////////////////////////////////////////////////////////////
    /// Verify that the values using tensor parallel and vanilla are the same
    /// //
    ////////////////////////////////////////////////////////////////////////////
    se.getHostValue(tParallelGrads[0])
        .assertAllClose(se.getHostValue(b_grads[0]), 1e-5, 1e-5);

    se.getHostValue(tParallelGrads[1])
        .assertAllClose(se.getHostValue(b_grads[1]), 1e-5, 1e-5);

    sgTensorParallel.append(std::cout);
    std::cout << std::endl;
  };

  compileRunVerify();

  auto redOps = g.opIds<ReduceAcrossReplicas>(sgTensorParallel);
  if (redOps.size() != 2) {
    throw poprithms::test::error(
        "Expected 2 reduction ops (forwards one, and its grad)");
  }

  removeRedundantReplicaReductions(sgTensorParallel);

  // Confirm that only one for replica reductions remains.
  redOps = g.opIds<ReduceAcrossReplicas>(sgTensorParallel);
  if (redOps.size() != 1) {
    throw poprithms::test::error("Expected just the forward reduction op to "
                                 "remain after the transfor");
  }

  // Confirm that the graph is still valid, we haven't done something stupid
  // in the transform.
  g.verifyValid();

  // Check that the numerics are the same.
  compileRunVerify();
}
} // namespace

int main() {
  dataReplication0();
  tensorParallel0();
  return 0;
}
