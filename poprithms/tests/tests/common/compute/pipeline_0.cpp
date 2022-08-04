// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <poprithms/common/compute/autodiff/autodiffer.hpp>
#include <poprithms/common/compute/autodiff/automaticquerier.hpp>
#include <poprithms/common/compute/memoryaliasmapper.hpp>
#include <poprithms/common/compute/ops/binaryelementwise.hpp>
#include <poprithms/common/compute/ops/dynamic.hpp>
#include <poprithms/common/compute/ops/init.hpp>
#include <poprithms/common/compute/ops/interdevicecopy.hpp>
#include <poprithms/common/compute/pipeline.hpp>
#include <poprithms/common/compute/prune/pruner.hpp>
#include <poprithms/common/compute/simexecutable.hpp>
#include <poprithms/common/compute/slickgraph.hpp>
#include <poprithms/common/compute/subgraph.hpp>
#include <poprithms/common/compute/tensor.hpp>
#include <poprithms/common/compute/testutil/finitedifference.hpp>
#include <poprithms/common/multiout/traversal.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/program/pipeline/objective.hpp>
#include <poprithms/program/pipeline/pipeline.hpp>
#include <poprithms/program/prune/prune.hpp>
#include <poprithms/util/typedinteger.hpp>

namespace {
using poprithms::program::pipeline::PipelineStage;
using poprithms::program::pipeline::PipelineStages;
using namespace poprithms::common::compute;

// Example:
//
// out = input + sqrt(|input|):
//
//
//  stage:  |  0   |   0  |   1   |   2   |
//  device: |  0   |   0  |   1   |   0   |
//          |      |      |       |       |
//  op:     |input -> abs -> sqrt -> add --> output.
//             |                      |
//             v                      ^
//             |                      |
//             +----------->----------+
//
// nToAccumulate : 10
// streamingIns  : input (first op)
// toAccumulate  : add (final op)
//
//  cycle       stages run        phase
// ---------    ----------        -----
//    0           0               ramp up
//    1           0 1             ramp up
//    2           0 1 2           repeat (full pipeline)
//        .
//        .
//        .
//    9           0 1 2           repeat (full pipeline)
//    10            1 2           ramp down
//    11              2           ramp down
//

// The logging for the unpipelined sub-graph looks like this:
//
// >  Entry  Shape  Graph              Device     Type
// >  -----  -----  -----              ------     ----
// >  *      ()     unpipelined(id=0)  Ipu(id=1)  float32
// >
// >  OpId  OpType   InTensors
// >  ----  ------   ---------
// >  0     VarInit  ()
// >  1     Abs      ((op=0))
// >  2     Sqrt     ((op=1))
// >  3     Add      ((op=0),(op=2))
//
// The pipelined version of this has 8 sub-graphs which work together to
// perform the the accumulation of the Add output for nToAccumulate=10 samples
// (10 cycles). They are described below

// 1) sg-initialize.
//    - initializes the accumulator tensor.
//    - sets the cycle index to zero.
//    - runs ops which don't depend on streamed inputs (none in this
//      example).
//
//
// >  Entry  Shape  Graph
// >  -----  -----  -----
// >  *      ()     sg-initialize(id=4)
//
// >  OpId Name              OpType           InTensors  Device    Type
// >  ---- ----              ------           ---------  ------    ----
// >  40                     RefFrom((op=35)) ()         Ipu(id=2) float32
// >  41   init accumulator  Fill_(0.000000)  ((op=40))  Ipu(id=2) float32
// >  42                     RefFrom((op=4))  ()         Ipu(id=2) unsigned32
// >  43   init index        Fill_(0)         ((op=42))  Ipu(id=2) unsigned32
// >  44                     RefFrom((op=5))  ()         Ipu(id=3) unsigned32
// >  45   init index        Fill_(0)         ((op=44))  Ipu(id=3) unsigned32

// 2, 3, 4) sg-stage-0, sg-stage-1, sg-stage-2.
//          - these sub-graphs perform the computation of each stage.
//
// clang-format off
//
// >  Entry  Graph             Device     Type
// >  -----  -----             ------     ----
// >  *      sg-stage-0(id=5)  Ipu(id=2)  float32
// >  
// >  OpId  Name               OpType   InTensors  Shape  IsRootOf
// >  ----  ----               ------   ---------  -----  --------
// >  6     clone of (opId=0)  VarInit  ()         ()     ((op=19))
// >  7     clone of (opId=1)  Abs      ((op=6))   ()     ((op=27))
// >  8     stash of (op=0)    VarInit  ()         (1)    ((op=9))
// >  
// >  
// >  
// >  OpId  Name                 OpType   InTensors  Shape  Graph             Device     IsRootOf   Type
// >  ----  ----                 ------   ---------  -----  -----             ------     --------   ----
// >  25    restore destination  VarInit  ()         ()     sg-stage-1(id=6)  Ipu(id=3)  ((op=26))  float32
// >  29    clone of (opId=2)    Sqrt     ((op=25))  ()     sg-stage-1(id=6)  Ipu(id=3)  ((op=32))  float32
// >  
// >  
// >  
// >  Entry  Shape  Graph             Device
// >  -----  -----  -----             ------
// >  *      ()     sg-stage-2(id=7)  Ipu(id=2)
// >  
// >  OpId  Name                 OpType            InTensors          IsRootOf   Type
// >  ----  ----                 ------            ---------          --------   ----
// >  18                         RefFrom((op=17))  ()                 ()         float32
// >  30    restore destination  VarInit           ()                 ((op=31))  float32
// >  34    clone of (opId=3)    Add               ((op=18),(op=30))  ()         float32
// >  35    accumulator          VarInit           ()                 ((op=40))  float32
// >  36                         RefFrom((op=4))   ()                 ()         unsigned32
// >  37                         ConstInit(2)      ()                 ()         unsigned32
// >  38                         Sub               ((op=36),(op=37))  ()         unsigned32
// >  39    accumulate           Add_              ((op=35),(op=34))  ()         float32
//
// clang-format on

// 5) sg-copy.
//   - copies tensors between devices.
//   - This is run between stage runs (and not interleaved).
//
// >  Entry  Shape  Graph          Type
// >  -----  -----  -----          ----
// >  *      ()     sg-copy(id=1)  float32
// >
// >  OpId  Name            OpType            InTensors          Device
// >  ----  ----            ------            ---------          ------
// >  26                    RefFrom((op=25))  ()                 Ipu(id=3)
// >  27                    RefFrom((op=7))   ()                 Ipu(id=2)
// >  28    dev:2 -> dev:3  CopyFrom_         ((op=26),(op=27))  Ipu(id=3)
// >  31                    RefFrom((op=30))  ()                 Ipu(id=2)
// >  32                    RefFrom((op=29))  ()                 Ipu(id=3)
// >  33    dev:3 -> dev:2  CopyFrom_         ((op=31),(op=32))  Ipu(id=2)

// clang-format off
// 6) sg-stash-restore-tick.
//   - stashes tensors for future use, 
//   - restores tensors from earlier cycles for immediate use, 
//   - increments cycle counters.
//
// Entry  Graph
// -----  -----
// *      sg-stash-restore-tick(id=3)
// 
// OpId  Name             OpType                         InTensors                 Shape  Device     IsRootOf           Type
// ----  ----             ------                         ---------                 -----  ------     --------           ----
// 4                      VarInit                        ()                        ()     Ipu(id=2)  ((op=36),(op=42))  unsigned32
// 5                      VarInit                        ()                        ()     Ipu(id=3)  ((op=44))          unsigned32
// 9                      RefFrom((op=8))                ()                        (1)    Ipu(id=2)  ()                 float32
// 10                     ConstInit(0)                   ()                        ()     Ipu(id=2)  ()                 unsigned32
// 11                     Add                            ((op=4),(op=10))          ()     Ipu(id=2)  ()                 unsigned32
// 12                     ConstInit(1)                   ()                        ()     Ipu(id=2)  ()                 unsigned32
// 13    restore index    Remainder                      ((op=11),(op=12))         ()     Ipu(id=2)  ()                 unsigned32
// 14                     Reshape_                       ((op=13))                 (1,1)  Ipu(id=2)  ()                 unsigned32
// 15                     VarInit                        ()                        (1,1)  Ipu(id=2)  ()                 float32
// 16                     DynamicMultiSlice_(dims=(0))   ((op=9),(op=15),(op=14))  (1,1)  Ipu(id=2)  ()                 float32
// 17    restore          Reshape_                       ((op=16))                 ()     Ipu(id=2)  ((op=18))          float32
// 19                     RefFrom((op=6))                ()                        ()     Ipu(id=2)  ()                 float32
// 20                     ConstInit(1)                   ()                        ()     Ipu(id=2)  ()                 unsigned32
// 21                     Remainder                      ((op=4),(op=20))          ()     Ipu(id=2)  ()                 unsigned32
// 22                     Reshape_                       ((op=19))                 (1,1)  Ipu(id=2)  ()                 float32
// 23                     Reshape_                       ((op=21))                 (1,1)  Ipu(id=2)  ()                 unsigned32
// 24                     DynamicMultiUpdate_(dims=(0))  ((op=9),(op=22),(op=23))  (1)    Ipu(id=2)  ()                 float32
// 46                     ConstInit(1)                   ()                        ()     Ipu(id=2)  ()                 unsigned32
// 47    increment index  Add_                           ((op=4),(op=46))          ()     Ipu(id=2)  ()                 unsigned32
// 48                     ConstInit(1)                   ()                        ()     Ipu(id=3)  ()                 unsigned32
// 49    increment index  Add_                           ((op=5),(op=48))          ()     Ipu(id=3)  ()                 unsigned32
// clang-format on

// 7) sg-full-pipe.
//   - Calls the following in sequence:
//     1) all stage sub-graphs (sg-stage-2, sg-stage-1, sg-stage-0),
//     2) sg-stash-restore-tick,
//     3) sg-copy.
//
// Entry  InTensors  Graph
// -----  ---------  -----
// *      ()         sg-full-pipe(id=8)
//
// OpId  Name                       OpType          NonDataIns
// ----  ----                       ------          ----------
// 58    full pipeline stage 2      Call(callee=7)  ()
// 59    full pipeline stage 1      Call(callee=6)  (58)
// 60    full pipeline stage 0      Call(callee=5)  (59)
// 61    stash, restore, increment  Call(callee=3)  (60)
// 62    inter-ipu copies           Call(callee=1)  (61)

// 8) sg-main.
//   - ramp-up calls
//   - repeat sg-full-pipe
//   - ramp-down calls.
//
// Entry  InTensors  Graph
// -----  ---------  -----
// *      ()         sg-main(id=2)
//
// OpId  Name                       OpType                      NonDataIns
// ----  ----                       ------                      ----------
// 50    initialization             Call(callee=4)              ()
// 51    ramp up stage 0            Call(callee=5)              (50)
// 52    stash, restore, increment  Call(callee=3)              (51)
// 53    inter device copies        Call(callee=1)              (52)
// 54    ramp up stage 1            Call(callee=6)              (53)
// 55    ramp up stage 0            Call(callee=5)              (54)
// 56    stash, restore, increment  Call(callee=3)              (55)
// 57    inter device copies        Call(callee=1)              (56)
// 63    full pipeline repeat       Repeat(id=8,repeatCount=8)  (57)
// 64    ramp down stage 2          Call(callee=7)              (63)
// 65    ramp down stage 1          Call(callee=6)              (64)
// 66    stash, restore, increment  Call(callee=3)              (65)
// 67    inter-ipu copies           Call(callee=1)              (66)
// 68    ramp down stage 2          Call(callee=7)              (67)
// 69    stash, restore, increment  Call(callee=3)              (68)
// 70    inter-ipu copies           Call(callee=1)              (69)
void testObjectiveExample() {

  SlickGraph g(400, ReplicationFactor::create(1));
  auto parts = g.partition(g.rootIpu(), 4);
  auto sg0   = g.createSubGraph("unpipelined");
  auto in0   = sg0.rootIpuFloat32Variable({});
  auto abso  = in0.abs();
  auto s     = abso.sqrt();
  auto addy  = in0 + s;

  AcclTypedObjective obj(
      {{in0.opId(), 0}, {abso.opId(), 0}, {s.opId(), 1}, {addy.opId(), 2}},

      {parts[0], parts[1], parts[0]},
      10,
      {addy.id()},
      {in0.id()});

  poprithms::common::compute::Pipeline pip(g, sg0, obj);

  for (uint64_t i = 0; i < g.nSubGraphs(); ++i) {
    std::cout << "\n\n";
    SubGraph(SubGraphId::createSubGraphId(i), g).append(std::cout);
    std::cout << std::endl;
  }
} // namespace

void test(SlickGraph &g,
          const SubGraph &sg0,
          const TensorIds &streamIns,
          const std::map<TensorId, HostTensor> &initialValues,
          const std::map<TensorId, HostTensor> &expectedAccumulations,
          uint64_t nStages,
          const std::vector<int> &stageDevIds,
          int64_t nSamples,
          PipelineAcclType acclType = PipelineAcclType::Sum) {

  TensorIds toAccumulate;
  for (auto x : expectedAccumulations) {
    toAccumulate.push_back(x.first);
  }

  for (auto vi : sg0.varInitIds()) {
    auto found = initialValues.find({vi, 0});
    if (found == initialValues.cend()) {
      throw poprithms::test::error("No value provided for var-init " +
                                   g.computeOp(vi).str());
    }
    auto shp = found->second.shape();
    if (std::find(streamIns.cbegin(), streamIns.cend(), TensorId{vi, 0}) !=
        streamIns.cend()) {
      if (shp != Shape{nSamples, 1}) {
        throw poprithms::test::error(
            "Shape of initial value for stream tensor is incorrect");
      }
    } else {
      if (shp.rank_i64() != 0) {
        throw poprithms::test::error(
            "Shape of initial value for non-stream tensor is incorrect");
      }
    }
  }

  // All tensors with an initial value in the map initialValues which are
  // not in streamIns are assumed to be non-stream inputs (weights).
  TensorIds nonStreamIns;
  for (const auto &[a, b] : initialValues) {
    if (std::find(streamIns.cbegin(), streamIns.cend(), a) ==
        streamIns.cend()) {
      nonStreamIns.push_back(a);
    }
  }

  const auto completeIpu = g.rootIpu();
  const auto partita     = g.partition(completeIpu, nStages);
  DeviceIds virtualGraphs;
  for (auto i : stageDevIds) {
    virtualGraphs.push_back(partita.at(i));
  }
  const auto fullSchedule = g.vanillaSchedule();

  // We split the ops evenly over the pipeline stages.
  const auto nOps = fullSchedule.size();
  std::map<OpId, PipelineStage> opToStage;
  for (uint64_t ps = 0; ps < nStages; ++ps) {
    const auto start = (nOps * ps / nStages);
    const auto end   = (nOps * (ps + 1) / nStages);
    for (auto i = start; i < end; ++i) {
      opToStage.insert({fullSchedule[i], ps});
    }
  }

  AcclTypedObjective obj(opToStage,
                         virtualGraphs,
                         nSamples,
                         toAccumulate,
                         {toAccumulate.size(), acclType},
                         streamIns);

  poprithms::common::compute::Pipeline pip(g, sg0, obj);

  std::map<TensorId, Tensor> streamInHostTensors;
  for (auto in0 : streamIns) {
    auto tHost = SubGraph(pip.stageSubGraph(in0.opId()), g)
                     .hostFloat32Variable({nSamples, 1});
    streamInHostTensors.insert({in0, tHost});

    Tensor toReplace(pip.getInStage(in0), &g);
    auto dataOnIpu = tHost.hostToIpu(toReplace.deviceId());
    g.removeOp(toReplace.opId(), {dataOnIpu.id()}, "wiring up to host");
  }

  for (auto nsi : nonStreamIns) {
    g.setInitialValue(pip.getInStage(nsi), 0, initialValues.at(nsi));
  }

  std::map<TensorId, Tensor> streamOutHostDatas;
  for (auto toStreamBack : toAccumulate) {

    auto backOnHost = Tensor(pip.accumulatorInStage(toStreamBack), &g)
                          .refTo_(pip.mainPipeline())
                          .ipuToHost(1);

    streamOutHostDatas.insert({toStreamBack, backOnHost});
  }

  g.verifyValid();
  g.setRunnable({pip.mainPipeline()});

  SimExecutable se(g);
  for (auto id : streamIns) {
    se.setHostValue(streamInHostTensors.at(id), initialValues.at(id));
  }
  se.run(pip.mainPipeline());

  for (auto x : toAccumulate) {
    se.getHostValue(streamOutHostDatas.at(x))
        .assertAllClose(
            expectedAccumulations.at(x),
            1e-5,
            1e-5,
            "comparing the pipelined model to a simple computation");
  }

  for (auto opId : g.opIds()) {
    using namespace poprithms::common::compute;
    if (g.dynamicCast<CopyFrom_>(opId) ||
        g.dynamicCast<CopyBetweenHostAndIpu_>(opId)) {
    } else {
      auto &&o = g.computeOp(opId);
      std::set<DeviceId> devIds;
      for (auto x : {o.inDeviceIds(), o.outDeviceIds()}) {
        for (auto y : x) {
          devIds.insert(y);
        }
      }
      if (devIds.size() > 1) {
        std::ostringstream oss;
        oss << "The op " << o << " has " << o.inDeviceIds()
            << " input devices and " << o.outDeviceIds()
            << " output devices. ";
        throw poprithms::test::error(oss.str());
      }
    }
  }
}

// First input is streamed input, second input is not (its a weight. constant
// for whole process).
template <typename ModelWith2Ins>
void test2ins(ModelWith2Ins &&chain,
              const std::vector<int> &devIds,
              int64_t nSamples,
              PipelineAcclType acclType = PipelineAcclType::Sum) {

  auto nStages = devIds.size();

  // 10 tiles per stage.
  SlickGraph g(nStages * 10, ReplicationFactor::create(1));
  auto sg0 = g.createSubGraph("sg0");

  auto tData0   = sg0.variable(DType::Float32, {}, g.rootIpu());
  auto tWeight0 = tData0.variable();

  auto rs0         = 1010 + nSamples + devIds.size();
  auto hostData0   = HostTensor::uniformFloat32(-1, 1, {nSamples, 1}, rs0);
  auto hostWeight0 = HostTensor::uniformFloat32(-1, 1, {}, rs0 + 100);

  // Grow the unpipeline graph:
  auto outs = chain(tData0, tWeight0);

  // Get the expected numerical values:
  auto expectedOuts = chain(hostData0, hostWeight0);
  std::map<TensorId, HostTensor> expectedOut;
  for (uint64_t i = 0; i < outs.size(); ++i) {

    switch (acclType) {

    case PipelineAcclType::Max: {
      expectedOut.insert(
          {outs[i],
           expectedOuts[i].expand({nSamples, 1}).reduceMax(Shape{})});
      break;
    }
    case PipelineAcclType::Sum: {
      expectedOut.insert(
          {outs[i],
           expectedOuts[i].expand({nSamples, 1}).reduceSum(Shape{})});
      break;
    }

    case PipelineAcclType::RunningMean: {
      expectedOut.insert({outs[i],
                          expectedOuts[i]
                              .expand({nSamples, 1})
                              .reduceSum(Shape{})
                              .div(nSamples)});
      break;
    }
    }
  }

  test(g,
       sg0,
       {tData0.id()},
       {{tData0, hostData0}, {tWeight0, hostWeight0}},
       expectedOut,
       nStages,
       devIds,
       nSamples,
       acclType);
}

template <typename ModelWith2Ins>
void test2ins(ModelWith2Ins &&chain,
              uint64_t nStages,
              int64_t nSamples,
              PipelineAcclType acclType = PipelineAcclType::Sum) {
  std::vector<int> devs(nStages);
  std::iota(devs.begin(), devs.end(), 0);
  return test2ins(chain, devs, nSamples, acclType);
}

template <typename T> std::vector<T> chainx(T in0, T in1) {
  auto t0 = in1 + in0.pow(2).sqrt().mul(in0).div(in0).add(1);
  auto t1 = in1 + in0.mul(0.5).add(t0).sub(t0).add(in0).mul(0.5);
  auto t2 = in1 + t1.add(t0).sub(in0);
  return {t0, t1, t2};
}

template <typename T> std::vector<T> chainy(T data0, T weight0) {
  auto a = data0 + weight0;
  auto b = a.relu().add(1).sqrt().mul(a).sub(2).div(a.abs().add(1));
  auto c = b.sin().cos();
  (void)c;
  (void)data0;
  return {weight0, data0, a, b};
}

template <typename T> std::vector<T> chainz(T data0, T weight0) {
  auto a = (data0 + weight0.abs().sqrt()).abs().sqrt();
  return {a, weight0};
}

template <typename T> std::vector<T> chainp(T data0, T weight0) {
  auto a = weight0.abs().sqrt();
  auto b = (data0 + weight0 + a).abs().sqrt();
  auto c = (data0 + weight0 + b + a).abs().sqrt();
  auto d = (data0 + weight0.relu() + c + b + a).abs().sqrt();
  return {weight0, data0, a, d, b};
}

void test0() {
  SlickGraph g(500, ReplicationFactor::create(1));
  auto parts = g.partition(g.rootIpu(), 5);
  auto sg0   = g.createSubGraph("sg0");
  auto in0   = sg0.rootIpuFloat32Variable({});
  auto a0    = in0.sin();
  auto a1    = a0.sin();
  auto a2    = a1 + in0;

  AcclTypedObjective obj(
      {{in0.opId(), 0}, {a0.opId(), 1}, {a1.opId(), 2}, {a2.opId(), 3}},
      parts,
      10,
      {a2.id()},
      {in0.id()});

  poprithms::common::compute::Pipeline pip(g, sg0, obj);

  // a2, in pipeline stage 3.
  // a2 uses in0, in pipeline stage 0.
  // We therefore expect a stash of size 2.
  int64_t expectedSize = 2;

  auto mSlices = g.opIds<DynamicMultiSlice_>();
  if (mSlices.size() != 1) {
    throw poprithms::test::error(
        "Expected exactly 1 dynamic slice (stash 'pop')");
  }
  auto ms = g.dynamicCast<DynamicMultiSlice_>(mSlices[0]);
  if (ms->sliceShape().nelms() * expectedSize !=
      ms->sliceableShape().nelms()) {
    throw poprithms::test::error(
        "Expected a stash of size 2 for pipeline stage 3 ahead");
  }
}

void test1() {

  SlickGraph g(400, ReplicationFactor::create(1));
  auto parts   = g.partition(g.rootIpu(), 4);
  auto sg0     = g.createSubGraph("sg0");
  auto in0     = sg0.rootIpuFloat32Variable({});
  auto weights = in0.variable().name("weights");
  auto a0      = weights * in0;
  auto a1      = a0.sin();
  auto a2      = a1.sin();
  auto a3      = a2 + weights;

  AcclTypedObjective obj({{in0.opId(), 0},
                          {weights.opId(), 0},
                          {a0.opId(), 0},
                          {a1.opId(), 1},
                          {a2.opId(), 2},
                          {a3.opId(), 3}},
                         {parts[0], parts[1], parts[2], parts[0]},
                         10,
                         {a3.id()},
                         {in0.id()});

  poprithms::common::compute::Pipeline pip(g, sg0, obj);

  // The weight tensor in the pipelined model.
  Tensor weightsInPipeline(pip.getInStage(weights), &g);
  weightsInPipeline.name("weightsInPipeline");
  MemoryAliasMapper mam(g, {weightsInPipeline});
  auto allRefsToWeights = mam.aliases({weightsInPipeline.id()});

  // The only consumers of the weights should be the add and the mul.
  for (auto x : allRefsToWeights) {
    for (auto c : g.consumptionIds(x)) {
      if (!g.dynamicCast<Mul>(c.opId()) && !g.dynamicCast<Add>(c.opId())) {
        std::ostringstream oss;
        oss << "The consumers of the weights (which are unchanging) are all "
               "on the same device. The consumers are (1) a mul and (2) an "
               "add. But it appears like the op "
            << g.computeOp(c.opId())
            << " also consumes the weights, which seems incorrect";
        throw poprithms::test::error(oss.str());
      }
    }
  }

  // a2, in pipeline stage 3.
}

void test3() {
  SlickGraph g(60, ReplicationFactor::create(1));
  auto parts = g.partition(g.rootIpu(), 2);
  auto sg0   = g.createSubGraph("sg0");
  auto in0   = sg0.rootIpuFloat32Variable({});
  auto a1    = in0.sin();

  AcclTypedObjective obj({{in0.opId(), 1}, {a1.opId(), 0}},
                         {parts[0], parts[1], parts[0]},
                         10,
                         {},
                         {});

  bool caught{false};
  try {
    poprithms::common::compute::Pipeline pip(g, sg0, obj);
  } catch (const poprithms::error::error &) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error("Failed to catch error where consumer is is "
                                 "earlier pipeline stage than producer");
  }
}

void test2() {

  SlickGraph g(60, ReplicationFactor::create(1));
  auto parts = g.partition(g.rootIpu(), 3);
  auto sg0   = g.createSubGraph("sg0");

  // stage 0, device 0
  auto in0 = sg0.rootIpuFloat32Variable({});
  auto a1  = in0.sin();

  // stage 1, device 1
  auto a2 = a1.sin();

  // stage 2, device 2
  auto a3 = in0 + a2;

  AcclTypedObjective obj(
      {{in0.opId(), 0}, {a1.opId(), 0}, {a2.opId(), 1}, {a3.opId(), 2}},
      {parts[0], parts[1], parts[0]},
      10,
      {},
      {in0.id()});

  poprithms::common::compute::Pipeline pip(g, sg0, obj);

  auto initIds = g.opIds<VarInit>();

  uint64_t nVarInitsOnDevPart0{0};
  for (auto x : initIds) {
    if (!g.isFixedPoint({x, 0})) {
      if (g.deviceId({x, 0}) == parts[0]) {
        nVarInitsOnDevPart0 += g.nelms({x, 0});
      }
    }
  }

  if (nVarInitsOnDevPart0 != 4) {
    std::ostringstream oss;
    oss << "On this device there is (1) input to stage 0 (2) stash (3) "
           "restore and (4) input to stage 2 from stage (1). Should be more "
           "than 4, but it is "
        << nVarInitsOnDevPart0;
    throw poprithms::test::error(oss.str());
  }
}

} // namespace

int main() {

  auto SUM  = PipelineAcclType::Sum;
  auto MEAN = PipelineAcclType::RunningMean;
  auto MAX  = PipelineAcclType::Max;

  // numerical tests:
  test2ins([](auto a, auto b) { return chainx(a, b); }, 5, 8, SUM);
  test2ins([](auto a, auto b) { return chainx(b, a); }, 4, 5, MEAN);
  test2ins([](auto a, auto b) { return chainx(a, b); }, {1, 0, 2, 0}, 8, MAX);
  test2ins([](auto a, auto b) { return chainx(b, a); }, {2, 1, 0, 0}, 5, SUM);
  test2ins([](auto a, auto b) { return chainx(b, a); }, {1, 1, 1}, 5, MEAN);
  test2ins([](auto a, auto b) { return chainy(a, b); }, 4, 9, SUM);
  test2ins([](auto a, auto b) { return chainy(a, b); }, 9, 9, MAX);
  test2ins([](auto a, auto b) { return chainy(a, b); }, {0, 1, 2, 0}, 9, SUM);
  test2ins([](auto a, auto b) { return chainz(a, b); }, {0, 1, 2, 3}, 9, MAX);

  test2ins(
      [](auto a, auto b) { return chainy(a, b); }, {0, 1, 0, 1, 4}, 9, MEAN);
  test2ins(
      [](auto a, auto b) { return chainp(a, b); }, {0, 1, 2, 0}, 4, MEAN);
  test2ins(
      [](auto a, auto b) { return chainz(a, b); }, {0, 1, 0, 1, 0}, 9, MEAN);

  // tests that there aren't too many copies, stashes aren't too big, etc.
  test0();
  test1();
  test2();
  test3();
  testObjectiveExample();
}
