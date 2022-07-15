// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <numeric>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/ops/binaryelementwise.hpp>
#include <poprithms/common/compute/ops/init.hpp>
#include <poprithms/common/compute/ops/unaryelementwise.hpp>
#include <poprithms/common/compute/ops/withcallees.hpp>
#include <poprithms/common/compute/prune/pruner.hpp>
#include <poprithms/common/compute/slickgraph.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/program/prune/prune.hpp>

namespace {

using namespace poprithms::common::compute;

void testPreserveHostTensors0() {
  SlickGraph m(1000, ReplicationFactor::create(1));

  // call op on ipu:
  //
  // in0 --+
  //       +--- add --mul ------+
  //       |           |        |  ... -> out0
  // in1 --+       constant     |
  //  |                        mul -----> out1
  //  +------>---sin ------->---+
  //
  auto sg1  = m.createSubGraph("sg1");
  auto in0  = sg1.variable(DType::Int32, {5}, m.rootIpu());
  auto in1  = in0.variable();
  auto out0 = (in0 + in1) * in0.constant(2.) + in0.relu();
  auto out1 = in0 * in1.sin();

  // host --> toIpu ---> abs -----------+
  //                      |             +-> call -> all internals copied out
  //                      +-----> copy -+
  //
  // only out0 is returned to host.
  auto sg0 = m.createSubGraph("sg0");
  auto x0  = sg0.hostInt32Variable({1, 1, 5}).hostToIpu(m.rootIpu()).abs();
  auto c0  = sg0.call(sg1, {{x0.copy(), in0}, {x0, in1}}, sg1.tensorIds());
  out0.dstInCaller(c0).ipuToHost(CircularBufferCount(1));

  m.setRunnable({sg0});
  Pruner::preserveHostTensors(m);

  // check that out1 is gone, that there are no sin ops, and that the call
  // only has 1 output.

  if (m.opIds<Sin>().size() != 0) {
    throw poprithms::test::error("Sin should have been pruned away");
  }

  if (m.isLive(out1.opId())) {
    throw poprithms::test::error("out1 doesn't lead to host, should be gone");
  }

  auto callOps = m.opIds<Call>();
  if (callOps.size() != 1) {
    throw poprithms::test::error("There  is only 1 call in this machine");
  }

  if (m.nOutTensors(callOps[0]) != 1) {
    throw poprithms::test::error(
        "only 1 output from the call should remain after pruning");
  }

  m.verifyValid();
}

void testCallIsPartitioned0() {

  SlickGraph m(1000, ReplicationFactor::create(1));

  /**
   *         +---- sin ---> x0
   *   in0 --+
   *         +---- cos ---> x1
   *
   *         */
  auto sg0 = m.createSubGraph("sg0");
  auto in0 = sg0.rootIpuFloat32Variable({5});
  auto x0  = in0.sin();
  auto x1  = in0.cos();

  /**
   *                   +--- call --+--
   * host --> to ipu --+           |
   *                   |           +-- add --> host
   *                   +--- call --+--
   * */
  auto sg1        = m.createSubGraph("sg1");
  auto in1        = sg1.hostFloat32Variable({1, 1, 5}).hostToIpu(m.rootIpu());
  auto c0         = sg1.call(sg0, {{in1, in0}}, {x0, x1});
  auto c1         = sg1.call(sg0, {{in1, in0}}, {x0, x1});
  auto backOnHost = (x0.dstInCaller(c0) + x1.dstInCaller(c1))
                        .ipuToHost(CircularBufferCount(1));
  (void)backOnHost;

  m.setRunnable({sg1});

  Pruner::preserveHostTensors(m);

  if (m.opIds<Sin>().empty() || m.opIds<Cos>().empty()) {
    throw poprithms::test::error("both sin and cos are on a path to host");
  }

  for (auto opId : m.opIds<Call>()) {
    if (m.nOutTensors(opId) != 1) {
      throw poprithms::test::error(
          "Calls only have 1 output on path to host");
    }
  }

  m.verifyValid();
}

void testPrune0() {
  SlickGraph m(1000, ReplicationFactor::create(1));

  // sg0:
  //    in0 -> sin -> out0
  auto sg0  = m.createSubGraph("sg0");
  auto in0  = sg0.rootIpuFloat32Variable({1});
  auto out0 = in0.cos();

  // sg1:
  //    in10 --> abs --> out10
  //    in11 --> sin --> out11
  //    in12 --> sg0 --> out12
  auto sg1   = m.createSubGraph("sg1");
  auto in10  = sg1.rootIpuFloat32Variable({1});
  auto in11  = in10.variable();
  auto in12  = in10.variable();
  auto out10 = in10.abs();
  auto out11 = in11.sin();
  auto c0    = sg1.call(sg0, {{in12, in0}}, {{out0}});
  auto out12 = out0.dstInCaller(c0);

  // sg2:
  //    in20 -> call(sg1) --> stream out12 to host.
  auto sg2  = m.createSubGraph("sg2");
  auto in20 = in10.variable(sg2);
  m.setInitialValue(in20, 0, HostTensor::float32({1}, {2.}));
  auto c1 = sg2.call(sg1, {{in20, in12}}, {out12, out10});

  out12.dstInCaller(c1).ipuToHost(CircularBufferCount(1));

  m.setRunnable({sg2});

  // expect:
  //   * in10, in11, out10, and out11 all to be removed.
  //   * the call to have 1 input and 1 output.

  Pruner::preserveHostTensors(m);

  if (m.isLive(in10.opId()) || m.isLive(in11.opId()) ||
      m.isLive(out10.opId()) || m.isLive(out11.opId())) {
    throw poprithms::test::error("Expected the sin and abs in sg1 (with "
                                 "inputs and outputs) to be removed");
  }

  auto calls = m.opIds<Call>();
  if (calls.size() != 2) {
    throw poprithms::test::error(
        "Expected 2 calls, one to sg1 from sg2 and one to sg0 from sg1");
  }

  if (m.nInTensors(c1) != 1 || m.nOutTensors(c1) != 1) {
    throw poprithms::test::error(
        "Expected c1 call to be pruned to have 1 input and 1 output");
  }

  m.verifyValid();
}

void testPruneRef0() {

  enum class ToRetain { Out11, Out10, Y };
  auto test = [](ToRetain toRetain) {
    /**
     * in10 -+                  +-> out10
     *       + --- call sg0  ---+
     * in11 -+                  +-> out11
     *                                |
     *                                +---> x = reference to sub-graph 2
     *
     * Care must be taken when pruning out10 : the ref of x must be updated.
     * */

    SlickGraph m;

    std::vector<SubGraph> sgs{m.createSubGraph("sg0"),
                              m.createSubGraph("sg1"),
                              m.createSubGraph("sg2")};

    auto ins0 = sgs[0].variables(DType::Int32, {{2}, {2}}, m.host());
    auto ins1 = sgs[1].variables(DType::Int32, {{2}, {2}}, m.host());

    // callee sub-graph:
    auto out00 = ins0[0] - ins0[1];
    auto out01 = ins0[0] + ins0[1];

    auto call0 =
        sgs[1].callAllOut(sgs[0], {{ins1[0], ins0[0]}, {ins1[1], ins0[1]}});

    auto out10 = out00.dstInCaller(call0);
    auto out11 = out01.dstInCaller(call0);
    auto x     = out11.refTo_(sgs[2]);
    auto y     = x.add(x.constant(1));

    m.setRunnable({sgs[1], sgs[2]});

    if (m.rootRef(x) != out11) {
      throw poprithms::test::error("Before pruning, the root ref is out11");
    }

    if (toRetain == ToRetain::Out10) {
      Pruner::prune(m, {out10});
      if (m.isLive(y.opId())) {
        throw poprithms::test::error(
            "y should be pruned if only out10 needed");
      }
    }

    if (toRetain == ToRetain::Out11) {
      Pruner::prune(m, {out11});
      // It is not obvious if y should be retained, so not testing for now.
    }

    if (toRetain == ToRetain::Y) {
      Pruner::prune(m, {y});
      if (m.rootRef(x).outIndex() != 0) {
        m.append(std::cout);
        std::cout << std::endl;
        throw poprithms::test::error(
            "After pruning, the root ref is output #0 "
            "(there should only be 1 output left)");
      }
    }
  };

  test(ToRetain::Out11);
  test(ToRetain::Out10);
  test(ToRetain::Y);
}

void testPruneCallInCall0() {

  SlickGraph m;

  //
  //  in00  -->  out00
  //  in01  -->  out01
  //
  auto sg0   = m.createSubGraph("sg0");
  auto in00  = sg0.hostInt32Variable({});
  auto in01  = in00.variable();
  auto out00 = in00.abs();
  auto out01 = in00.sin();

  //
  //  in10  --+          +-> out10
  //          +--> sg0 --+
  //  in11  --+          +-> out11
  //
  //  in12 ----------------> out12
  //
  auto sg1   = m.createSubGraph("sg1");
  auto in10  = in00.variable(sg1);
  auto in11  = in10.variable();
  auto in12  = in10.variable();
  auto c1    = sg1.callAllOut(sg0, {{in10, in00}, {in11, in01}});
  auto out10 = out00.dstInCaller(c1);
  auto out11 = out01.dstInCaller(c1);
  auto out12 = in12.sqrt();

  //
  //  in20  --+          +-> out20
  //          |          |
  //  in21  --+--- sg1 --+-> out21
  //          +          |
  //  in22 ---+          +-> out22

  auto sg2  = m.createSubGraph("sg2");
  auto in20 = in00.variable(sg2);
  auto in21 = in20.variable();
  auto in22 = in20.variable();
  auto c2   = sg2.callAllOut(sg1, {{in20, in10}, {in21, in11}, {in22, in12}});
  auto out20 = out10.dstInCaller(c2);
  auto out21 = out11.dstInCaller(c2);
  auto out22 = out12.dstInCaller(c2);

  for (Tensor t : {out20, out21, out22}) {
    auto m2 = m;
    m2.setRunnable({sg2});
    Pruner::prune(m2, {t});
    m2.verifyValid();

    if (t.id() == out20 || t.id() == out21) {
      if (m2.nTensors() != 6) {
        throw poprithms::test::error("Expect 6 tensors (3 in 3 out)");
      }
    }

    if (t.id() == out22) {
      if (m2.nTensors() != 4) {
        throw poprithms::test::error("Expect 4 tensors (2 in 2 out)");
      }
    }
  }
}

void testCopyToCallOut0() {
  SlickGraph m;

  auto sg0   = m.createSubGraph("sg0");
  auto in0a  = sg0.hostInt32Variable({3});
  auto in0b  = in0a.variable(Shape({5}));
  auto out0a = in0a.sin();
  auto out0b = in0b.cos();

  auto sg1   = m.createSubGraph("sg1");
  auto in1a  = sg1.hostInt32Variable({3});
  auto in1b  = in1a.variable(Shape({5}));
  auto op1   = sg1.callAllOut(sg0, {{in1a, in0a}, {in1b, in0b}});
  auto out1a = out0a.dstInCaller(op1);
  auto out1b = out0b.dstInCaller(op1);

  auto sg2  = m.createSubGraph("sg2");
  auto in2a = sg2.hostInt32Variable({3});
  auto in2b = in2a.variable(Shape({5}));

  // This is a very unconventional call, as the destinations in the callee are
  // outputs, not variable initializers.
  auto op2 = sg2.callAllOut(sg1, {{in2a, out1a}, {in2b, out1b}});

  (void)op2;

  m.setRunnable({sg2});

  Pruner::prune(m, {out1b.dstInCaller(op2)});

  if (m.opIds<Sin>().size() != 0) {
    throw poprithms::test::error(
        "The sin op should be removed, not on path to pruned");
  }
  m.verifyValid();
}

// This is a test for T63457.
void testRemoveOutput0() {

  SlickGraph m;

  auto sg0 = m.createSubGraph("sg0");
  auto x00 = sg0.hostInt32Variable({1, 2});
  auto x01 = sg0.hostFloat32Variable({3, 4});

  auto sg1 = m.createSubGraph("sg1");
  auto op1 = sg1.callAllOut(sg0, {});
  auto x10 = x00.dstInCaller(op1);
  auto x11 = x01.dstInCaller(op1);
  auto sub = x10.variable();

  auto sg2 = m.createSubGraph("sg2");
  auto x21 = x11.variable(sg2);
  auto op2 = sg2.callAllOut(sg1, {{x21, x11}});

  if (x21.dstsInCallee(m.callEvent(op2)).at(0).outIndex() != 1) {
    throw poprithms::test::error(
        "Problem case set up so that the output index of in copy is 1");
  }

  // We remove sub as an output, because if we don't then using it as a
  // replacement will mean it is the output at 2 indices: not supported.
  m.removeOutputs(
      op2, {sub.dstInCaller(op2).outIndex()}, OptionalTensorIds(1));
  m.removeOutputs(op1, {x10.outIndex()}, {sub.id()});
  m.verifyValid();

  if (x21.dstsInCallee(m.callEvent(op2)).at(0).outIndex() != 0) {
    throw poprithms::test::error("Failed to shift input index down");
  }

  auto subOutCopies = m.computeOp(sub.opId()).outCopies(sub.outIndex());
  if (subOutCopies.size() != 1) {
    throw poprithms::test::error("Sub is copied out once");
  }
  if (subOutCopies.at(0).caller() != op2) {
    throw poprithms::test::error("Sub is copied out of call op2");
  }

  m.verifyValid();

  (void)op2;
}

void testPruneMlMock0() {

  SlickGraph g;

  // program to do some kind of training.
  auto sg0      = g.createSubGraph("main");
  auto lr       = sg0.rootIpuFloat32Variable({});
  auto w0       = lr.variable({4, 4});
  auto dx       = w0.abs().sqrt();
  auto w0update = w0.add_(dx * lr);
  auto stat = w0update.reduceSum(Shape{}).ipuToHost(CircularBufferCount(1));

  // program to update learning rate.
  auto sg1    = g.createSubGraph("updateLr");
  auto lr0    = sg1.hostFloat32Variable({1, 1});
  auto lrNew0 = lr.refTo_(sg1).updateFromHost_(lr0);

  // The inplace power(2) will change the learning rate in sg0.
  // The inplace power(3) will not change the lr in sg0.
  auto lrNew = lrNew0.pow_(lrNew0.constant(2)).pow(lrNew0.constant(3));

  // program to reset weights.
  auto sg2              = g.createSubGraph("updateW0");
  auto w0h              = sg2.hostFloat32Variable({1, 1, 4, 4});
  auto w0updateFromHost = w0.refTo_(sg2).updateFromHost_(w0h);

  g.setRunnable({sg0, sg1, sg2});

  Pruner::preserveHostTensors(g);
  auto opIds = g.opIds<Pow_>();
  if (opIds.size() != 1) {
    throw poprithms::test::error("Expected just 1 pow_ to remain");
  }
  const auto expo =
      g.dynamicMutableCast<ConstInit>(g.inTensorId(opIds.at(0), 1).opId())
          ->value();

  if (expo.getInt32(0) != 2) {
    throw poprithms::test::error(
        "Expected the exponent of the remaining opwer op to be 2");
  }

  (void)stat;
  (void)lrNew;
  (void)w0updateFromHost;
}

void testPruneTrickyAliases0() {

  auto test = [](int64_t ub) {
    SlickGraph g;
    // Just a host->ipu->host.
    auto sg0   = g.createSubGraph("sg0");
    auto v0h   = sg0.hostInt32Variable({1, 1, 10});
    auto v0    = v0h.hostToIpu(g.rootIpu());
    auto vBack = v0.ipuToHost(CircularBufferCount(1));

    // Creates a reference to the ipu tensor in sg0, and potentially modifies
    // it. Modifies it if ub (below) is greater than 10. So we check if the
    // the cos_ which modifies it is removed when ub <= 10.
    auto sg1   = g.createSubGraph("sg1");
    auto v0ref = v0.refTo_(sg1);
    auto v1    = v0ref.variable();
    g.setInitialValue(v1, 0, HostTensor::int32(0).expand({10}));

    auto twoCats     = Tensor::concat_({v1, v0ref}, 0);
    auto v1sliceBack = twoCats.slice_(Dimension(0), 0, ub);
    v1sliceBack.cos_().ipuToHost(CircularBufferCount(1));
    g.setRunnable({sg0, sg1});
    Pruner::prune(g, {vBack});

    if (ub > 10) {
      if (g.opIds<Cos_>().size() != 1) {
        throw poprithms::test::error(
            "When ub > 10, the aliasing graph modifies the ipu tensor and so "
            "cos_ cannot be removed.");
      }
    } else {
      if (g.opIds<Cos_>().size() != 0) {
        throw poprithms::test::error("When ub <= 10, the aliasing graph does "
                                     "not modifiy the ipu tensor and so "
                                     "cos_ can be removed.");
      }
    }
  };

  // removed:
  test(10);

  // not removed:
  test(11);
}

void testNotRetainConstraints0() {

  SlickGraph g;
  auto sg0 = g.createSubGraph("sg0");
  auto in0 = sg0.variable(DType::Int16, Shape({3}), g.host());

  auto out0   = in0.sin().cos();
  auto inter1 = in0.abs().sqrt();
  auto out1   = inter1.add(inter1.constant(1)).sqrt();
  g.constraint(out1.opId(), out0.opId());

  g.setRunnable({sg0});

  Pruner::prune(g, {out0, inter1});

  for (auto opId : g.opIds()) {
    if (g.computeOp(opId).controlDependencyInOps().size() +
            g.computeOp(opId).controlDependencyOutOps().size() !=
        0) {
      throw poprithms::test::error(
          "Did not expect the control dep to be transferred.");
    }
  }
}
} // namespace

int main() {

  // prune with call op:
  testPreserveHostTensors0();
  testCallIsPartitioned0();
  testPrune0();
  testPruneRef0();
  testPruneCallInCall0();
  testCopyToCallOut0();
  testRemoveOutput0();
  testPruneMlMock0();
  testPruneMlMock0();
  testPruneTrickyAliases0();
  testPruneTrickyAliases0();
  testNotRetainConstraints0();

  return 0;
}
