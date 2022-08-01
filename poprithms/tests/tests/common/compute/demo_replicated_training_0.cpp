// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <poprithms/common/compute/autodiff/autodiffer.hpp>
#include <poprithms/common/compute/simexecutable.hpp>
#include <poprithms/common/compute/slickgraph.hpp>

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

int main() {

  using namespace poprithms::common::compute;

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
  auto dw0   = ir.tensor(dw0id).reduceSumAcrossReplicas();

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
  return 0;
}
