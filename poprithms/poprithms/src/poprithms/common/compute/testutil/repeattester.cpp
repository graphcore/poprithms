// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <memory>
#include <sstream>
#include <string>

#include <poprithms/common/compute/autodiff/autodiffer.hpp>
#include <poprithms/common/compute/autodiff/automaticquerier.hpp>
#include <poprithms/common/compute/slickgraph.hpp>
#include <poprithms/common/compute/testutil/finitedifference.hpp>
#include <poprithms/common/compute/testutil/repeattester.hpp>
#include <poprithms/error/error.hpp>

namespace poprithms {
namespace common {
namespace compute {
namespace testutil {

void PolyExecutableTester::noWeakVTables() {
  throw poprithms::error::error("testutil",
                                error::error::weakVTableMessage());
}

void RepeatTester::testLadder0() {

  /**
   *
   *  in0 (stacked) ---> +2 ---> out0
   *                              .
   *                              .
   *   ...................<........
   *   .        (carry)
   *   .
   *  in1 (flat) ---> *2 ---> out1
   *                           .
   *                           .
   *   .........................
   *   .        (carry)
   *   .
   *   V
   *   .
   *  in2 (flat) ---> pow(2) ---> out2
   *
   * */

  SlickGraph m;
  auto sg0  = m.createSubGraph("sg0");
  auto in0  = sg0.hostFloat64Variable({});
  auto in1  = in0.variable();
  auto in2  = in0.variable();
  auto out0 = in0 + in0.constant(2);
  auto out1 = in1 * in1.constant(2);
  auto out2 = in2 * in2;

  int64_t nReps{5};
  auto sg1  = m.createSubGraph("sg1");
  auto in1_ = in1.variable(sg1);
  auto in2_ = in2.variable(sg1);
  auto in0_ = in1_.variable(Shape{nReps});

  auto rpt =
      sg1.repeat(sg0,
                 nReps,
                 {{{in0_, in0}}},
                 {{{in1_, in1, out0}, {in2_, in2, out1}}},
                 {{{out0, IsStackedCopy::Yes}, {out2, IsStackedCopy::Yes}}});
  m.setRunnable({sg1});

  auto finale = out0.dstInCaller(rpt) + out2.dstInCaller(rpt);
  auto host0  = HostTensor::float64({nReps}, {2, 3, 4, 5, 6});

  setCompiledSlickGraph(m);

  cm().setHostValue(in0_, host0);
  cm().setHostValue(in1_, HostTensor::float64(12));
  cm().setHostValue(in2_, HostTensor::float64(13));
  cm().run(sg1);

  auto expOut0 = host0.add(2);
  auto expOut2 = HostTensor::concat(
      {HostTensor::float64(13 * 13).reshape({1}),
       HostTensor::float64(24 * 24).reshape({1}),
       host0.add(2).mul(2).pow(2).slice(Dimension(0), 0, 3)},
      0);

  cm().getHostValue(finale).assertAllEquivalent(expOut0 + expOut2);

  {
    bool caught{false};
    try {
      Autodiffer(m).backward(finale.reduceSum(), {in0_});
    } catch (const poprithms::error::error &) {
      caught = true;
    }
    if (!caught) {
      throw poprithms::test::error(
          "Failed to catch error with insufficient checkpointing");
    }
  }
}

void RepeatTester::testIsStackedOut() {

  SlickGraph m;
  auto sg0 = m.createSubGraph("sg0");
  auto x0  = sg0.hostInt32Variable({});
  auto x1  = x0.variable();
  auto x2  = x0.variable();
  auto x3  = x0.variable();

  auto sg1       = m.createSubGraph("sg1");
  auto phi       = sg1.repeat(sg0,
                        10,
                        {},
                        {},
                        {{{x0, IsStackedCopy::Yes},
                                {x1, IsStackedCopy::No},
                                {x2, IsStackedCopy::Yes}}});
  const auto rpt = m.dynamicCast<Repeat>(phi);

  localAssert(rpt->isStackedOut(x0.id()), "x0 is a stacked out");
  localAssert(!rpt->isStackedOut(x1.id()), "x1 is a flat out");
  localAssert(!rpt->isStackedOut(x3.id()), "x3 is a flat out");
  localAssert(rpt->isFlatOut(x1.id()), "x1 is a flat out");
  localAssert(!rpt->isFlatOut(x0.id()), "x0 is a stacked out");
}

void RepeatTester::testVisited0() {
  SlickGraph m;
  auto sg0 = m.createSubGraph("sg0");
  auto in0 = sg0.hostFloat32Variable({});
  auto a   = in0.relu();
  auto b   = a.sin();
  auto x   = b.sqrt();

  auto sg1 = m.createSubGraph("sg1");
  auto in1 = in0.variable(sg1);
  auto rpt =
      sg1.repeat(sg0, 3, {}, {{{in1, in0, x}}}, {{{x, IsStackedCopy::No}}});

  auto v = m.dynamicCast<Repeat>(rpt)->gradientPropagationVisits({0}, {0});
  for (auto t : {in0, a, b, x}) {
    if (std::find(v.cbegin(), v.cend(), t) == v.cend()) {
      std::ostringstream oss;
      oss << "The tensor t was visited, where t=" << t.id() << '.';
      throw poprithms::test::error(oss.str());
    }
  }
}

void RepeatTester::testStackedInput0() {

  SlickGraph m(1000, ReplicationFactor::create(1));

  auto sg0  = m.createSubGraph("sg0");
  auto in00 = sg0.hostInt32Variable({4});
  auto in01 = sg0.hostInt32Variable({4});
  auto out0 = in00 + in01;

  auto sg1 = m.createSubGraph("sg1");
  uint64_t repeatCount{3};
  auto in10 = in00.variable(sg1);
  auto in11 = in10.variable(in01.shape().prepend(repeatCount));

  // Copy the value of in10 to in00, then repeat:
  // -> Copy from the stacked tensor in11 to in01
  // -> perform addition (run sg0)
  // -> Copy the value of out0 back to in00
  // Finally, copy out0 out.
  auto r0 = sg1.repeat(sg0,
                       repeatCount,
                       {{{in11, in01}}},
                       {{{in10, in00, out0}}},
                       {{{out0, IsStackedCopy::No}}});

  auto outFinal = m.tensor({r0, 0}).copy();

  m.setRunnable({sg1});

  setCompiledSlickGraph(m);

  cm().setHostValue(in10, HostTensor::int32({4}, {0, 10, 100, 1000}));

  int64_t repeatCount_i64{static_cast<int64_t>(repeatCount)};

  // [[0 1 2 3] [4 5 6 7] [8 9 10 11]]
  cm().setHostValue(in11,
                    HostTensor::arangeInt32(0, repeatCount * 4, 1)
                        .reshape({repeatCount_i64, 4}));

  cm().run(sg1);
  auto expected = HostTensor::int32(
      {4}, {0 + 4 + 8, 10 + 1 + 5 + 9, 100 + 2 + 6 + 10, 1000 + 3 + 7 + 11});
  cm().getHostValue(outFinal).assertAllEquivalent(expected);
}

void RepeatTester::testRepeat0() {

  int64_t rFact{2};
  SlickGraph m(1000, ReplicationFactor::create(rFact));

  auto sg0 = m.createSubGraph("sg0");

  auto in00 = sg0.variable(DType::Int32, {}, m.rootIpu());
  auto in01 = in00.variable();
  auto x0   = in00 * in00.constant(2);
  auto out0 = x0 + in01 + in01.constant(-5);

  auto sg1            = m.createSubGraph("sg1");
  int64_t repeatCount = 3;
  auto in10host =
      sg1.variable(DType::Int32, {1, rFact, repeatCount}, m.host());
  auto in10 = in10host.hostToIpu(m.rootIpu());

  auto in11host = sg1.variable(DType::Int32, {1, rFact}, m.host());
  auto in11     = in11host.hostToIpu(m.rootIpu());

  auto rep =
      sg1.repeat(sg0,
                 repeatCount,
                 {{{in10, in00}}},
                 {{{in11, in01, out0}}},
                 {{{x0, IsStackedCopy::Yes}, {out0, IsStackedCopy::No}}});

  auto x0Host   = x0.dstInCaller(rep).ipuToHost(CircularBufferCount(1));
  auto out0Host = out0.dstInCaller(rep).ipuToHost(CircularBufferCount(1));

  m.setRunnable({sg1});

  setCompiledSlickGraph(m);

  cm().setHostValue(in10host,
                    HostTensor::int32({1, 2, 3}, {0, 1, 2, 3, 4, 5}));
  cm().setHostValue(in11host, HostTensor::int32({1, 2}, {10, 20}));

  // on replica 0:
  // in00 = (0,1,2) and in11 = 10.
  //
  //   x0   : 0 -> 2 -> 4
  //   out0 :   5 -> 2 -> 1
  //
  // on replica 1:
  // in00 = (3,4,5) and in11 = 20.
  //
  //   x0   : 6 -> 8 -> 10
  //   out0 :   21 -> 24 -> 29.

  cm().run({sg1});

  cm().getHostValue(x0Host).assertAllEquivalent(
      HostTensor::int32({2, 3}, {0, 2, 4, 6, 8, 10}));

  cm().getHostValue(out0Host).assertAllEquivalent(
      HostTensor::int32({2}, {1, 29}));
}

void RepeatTester::testGlobalPower0() {

  // training test #1.
  // repeat with a carry dependency, and no stacked inputs.

  /**
   * in0 --+
   *       +--- pow -- out0 --> carried copy back to in0
   * 1.5 --+
   * */
  SlickGraph m(1000, ReplicationFactor::create(1));
  auto sg0  = m.createSubGraph("sg0");
  auto in0  = sg0.hostFloat64Variable({});
  auto pwr  = in0.constant(1.5);
  auto out0 = in0.pow(pwr);

  /**
   * call sg0 3 times. So effectively get in1.pow(1.5).pow(1.5).pow(1.5)
   * */
  auto sg1 = m.createSubGraph("sg1");
  auto in1 = in0.variable(sg1);

  auto c0 =
      sg1.repeat(sg0,
                 3,
                 {},
                 // ins:
                 {{{in1, in0, out0}}},
                 // outputs:
                 {{{out0, IsStackedCopy::No}, {in0, IsStackedCopy::Yes}}});

  auto out1 = out0.dstInCaller(c0);
  auto dIn1 = m.tensor(Autodiffer(m).backward(out1, {in1})[0]).copy();
  m.setRunnable({sg1});

  setCompiledSlickGraph(m);
  cm().setHostValue(in1, HostTensor::float64(2.0));
  cm().run(sg1);

  // 1.5 * 1.5 * 1.5 = 3.375
  // d/dx (x^) = 3.375 * x^2.375
  // evaluated at x = 2 is 17.50733398778863
  cm().getHostValue(dIn1).assertAllClose(
      HostTensor::float64(17.50733398), 1e-3, 1e-3);
}

void RepeatTester::testShardedSinTrain0() {

  // training test #2.
  // repeat with no carry dependency, and 1 stacked input. This is like a
  // sharded matmul.

  SlickGraph m;
  auto sg0 = m.createSubGraph("sg0");
  auto in0 = sg0.hostFloat32Variable({2});
  auto out = (in0 + in0.constant(2)).sin().pow(in0.constant(3));

  // chain rule:
  // d/dx sin(x + 2)^3
  //     = 3*cos(x+2)*sin(x+2)^2.

  int64_t repeatCount{3};
  auto sg1 = m.createSubGraph("sg1");
  auto in1 = sg1.hostFloat32Variable({repeatCount, 2});
  auto rptOp =
      sg1.repeat(sg0,
                 repeatCount,
                 {{{in1, in0}}},
                 {},
                 // outs:
                 {{{in0, IsStackedCopy::Yes}, {out, IsStackedCopy::Yes}}});

  auto loss = out.dstInCaller(rptOp).reduceSum();
  auto dIn1 = m.tensor(Autodiffer(m).backward(loss, {in1})[0]).copy();

  m.setRunnable({sg1});

  setCompiledSlickGraph(m);
  auto hInVals =
      HostTensor::uniformFloat32(-1.0, 1.0, {repeatCount, 2}, 1011);
  cm().setHostValue(in1, hInVals);
  cm().run(sg1);

  // see chain rule calculation.
  auto p2             = hInVals.add(2);
  const auto expected = p2.cos() * (p2.sin().pow(2)).mul(3);
  cm().getHostValue(dIn1).assertAllClose(expected, 1e-3, 1e-3);
}

void RepeatTester::testRepeatCarriedPreStackedToLoss() {

  // what's interesting about this one is that the carried back tensor is on a
  // path to the stacked output.

  SlickGraph m;

  //       carried back
  //  <=====================
  // in0  ---> pow(2) ---> x0 ----+
  //  |                           +---> x1 ( == in0 * in0 + in0)
  //  +---------------------------+
  auto sg0 = m.createSubGraph("sg0");
  auto in0 = sg0.hostFloat64Variable({5});
  auto x0  = in0 * in0; // pow(in0.constant(2));
  auto x1  = x0 + in0;

  auto sg1    = m.createSubGraph("sg1");
  auto inMain = in0.variable(sg1);
  auto rpt    = sg1.repeat(sg0,
                        3,
                        {},
                        // ins:
                        {{{inMain, in0, x0}}},
                        // outs:
                        {{{x0, IsStackedCopy::No},
                             {x1, IsStackedCopy::Yes},
                             {in0, IsStackedCopy::Yes}}});

  auto loss = x1.dstInCaller(rpt).reduceSum(Shape({}));

  // the stacked outputs for x1 are  (for repeat count 3)
  // x^2 + x
  // x^4 + x^2
  // x^8 + x^4.
  //
  // grad of which is 8*x^7 + 8*x^3 + 4*x + 1 (for repeat count 3)
  auto init0 = HostTensor::float64({5}, {0, 1, -2, 3., 4});

  auto expected =
      init0.pow(7).mul(8) + init0.pow(3).mul(8) + init0.mul(4).add(1);

  auto dInMain = m.tensor(Autodiffer(m).backward(loss, {inMain})[0]).copy();

  if (dInMain.shape() != inMain.shape()) {
    throw poprithms::test::error("gradient has a different shape");
  }

  m.setRunnable({sg1});

  setCompiledSlickGraph(m);

  cm().setHostValue(inMain, init0);
  cm().run(sg1);

  cm().getHostValue(dInMain).assertAllEquivalent(expected);
}

void RepeatTester::testRepeatStackedToLossPreCarried() {

  // This test is like the preceding one, but now the carry and stacked out
  // are switched.

  SlickGraph m;

  auto sg0 = m.createSubGraph("sg0");
  auto in0 = sg0.hostFloat64Variable({5});
  auto x0  = in0 * in0;
  auto x1  = x0 + in0;

  auto sg1    = m.createSubGraph("sg1");
  auto inMain = in0.variable(sg1);
  auto rpt    = sg1.repeat(sg0,
                        3,
                        {},
                        {{{inMain, in0, x1}}},
                        {{{x1, IsStackedCopy::No},
                             {in0, IsStackedCopy::Yes},
                             {x0, IsStackedCopy::Yes}}});

  auto loss = x0.dstInCaller(rpt).reduceSum(Shape{}).copy();

  auto dInMain = m.tensor(Autodiffer(m).backward(loss, {inMain})[0]).copy();

  if (dInMain.shape() != inMain.shape()) {
    throw poprithms::test::error("gradient has a different shape");
  }

  m.setRunnable({sg1});

  bool canonicalizeRepeats = {false};

  setCompiledSlickGraph(m);

  // stacked stacked outputs for x0 are
  // a = x**2
  // b = (a+x)**2
  // c = (b+a+x)**2
  //
  // da = 2x
  // db = 2(a+x)(da+1)
  // dc = 2(b+a+x)(db+da+1)
  //
  // the gradient is a polynomial with integral coefficiencts, so we expect
  // exact match if input is integral.
  auto init0 = HostTensor::float64({5}, {0, 1, -2, 3., 4});

  auto a = init0.pow(2);
  auto b = (a + init0).pow(2);
  auto c = (b + a + init0).pow(2);

  auto grada    = init0.mul(2);
  auto gradb    = (a + init0).mul(grada.add(1)).mul(2);
  auto gradc    = (b + a + init0).mul(2).mul(gradb + grada.add(1));
  auto expected = grada + gradb + gradc;

  cm().setHostValue(inMain, init0);
  cm().run(sg1);
  expected.assertAllEquivalent(cm().getHostValue(dInMain));

  // Can also do a finite difference test (not necessary as already covered in
  // above exact test)
  if (!canonicalizeRepeats) {

    finiteDifferenceTest(
        cm(), loss, inMain, dInMain, {{inMain, init0}}, 1011, 1e-6);
  }
}

void RepeatTester::testOffPathCarriedIncr() {

  SlickGraph m;

  //         in1 ----+
  //  in0     |      |
  //   |     cast    |
  //   |      |    add1
  //   +--+---+      |
  //      |        carried to in1
  //     mul
  //      |
  //  carried to in0
  //
  auto sg0  = m.createSubGraph("sg0");
  auto in0  = sg0.hostFloat64Variable({});
  auto in1  = sg0.hostInt32Variable({});
  auto out0 = in0.mul(in1.to(DType::Float64));
  auto out1 = in1.add(in1.constant(1));

  auto sg1 = m.createSubGraph("sg1");
  auto in2 = in0.variable(sg1);
  auto in3 = sg1.constant(DType::Int32, {}, m.host());

  auto rpt = sg1.repeatAllOut(
      sg0, 3, {}, {{{in2, in0, out0}, {in3, in1, out1}}}, {out0, out1});

  auto loss = out0.dstInCaller(rpt).abs().reduceSum(Shape({}));

  auto grad = m.tensor(Autodiffer(m).backward(loss, {in2})[0]).copy();

  HostTensor in2host = HostTensor::float64(5);
  HostTensor in3host = HostTensor::int32(2);

  m.setRunnable({sg1});

  setCompiledSlickGraph(m);

  cm().setHostValue(in2, in2host);
  cm().setHostValue(in3, in3host);
  cm().run(sg1);

  // out = (x0 * 2) * 3 * 4.
  // grad = 24 * 1 = 24.

  cm().getHostValue(grad).assertAllEquivalent(HostTensor::float32(24));
}

void RepeatTester::testMultiInCall0() {

  SlickGraph m;

  auto sg0 = m.createSubGraph("sg0");
  auto in0 = sg0.hostFloat64Variable({});
  auto in1 = in0.variable();
  auto in2 = in0.variable();
  auto out = in0 * in1 * in2 + in0.constant(17);

  auto sg1  = m.createSubGraph("sg1");
  auto in3  = in0.variable(sg1);
  auto c0   = sg1.callAllOut(sg0, {{{in3, in0}, {in3, in1}, {in3, in2}}});
  auto loss = out.dstInCaller(c0);

  auto dIn3 = m.tensor(Autodiffer(m).backward(loss, {in3})[0]).copy();
  m.setRunnable({sg1});

  setCompiledSlickGraph(m);
  cm().setHostValue(in3, HostTensor::float64(5.));

  cm().run(sg1);

  // loss(x) = x^3 + 17.
  // dloss / dx = 3x^2.
  // @5 = 75.
  cm().getHostValue(dIn3).assertAllEquivalent(HostTensor::float64(75.));
}

void RepeatTester::testCrissCross0() {
  SlickGraph m;
  auto sg0 = m.createSubGraph("sg0");
  auto in0 = sg0.hostFloat64Variable({7});
  auto in1 = sg0.hostFloat64Variable({7});
  auto x0  = in0 * in0;
  auto x1  = in1.sin();

  auto sg1 = m.createSubGraph("sg1");
  auto in2 = in0.variable(sg1);
  auto in3 = in2.copy();
  (void)in3;
  auto rpt = sg1.repeatAllOut(
      sg0, 3, {}, {{{in2, in0, x1}, {in2, in1, x0}}}, {x1, x0});

  auto loss = (x0.dstInCaller(rpt) + x1.dstInCaller(rpt)).reduceSum(Shape{});

  auto dIn2 = m.tensor(Autodiffer(m).backward(loss, {in2})[0]);

  m.setRunnable({sg1});

  setCompiledSlickGraph(m);

  auto init0 = HostTensor::float64({7}, {0, 1.1, -3.1, 4., 0.1, 1.5, -0.8});

  finiteDifferenceTest(cm(), loss, in2, dIn2, {{in2, init0}}, 1011, 1e-5);
}

void RepeatTester::testProduct0() {
  SlickGraph m;
  auto sg0 = m.createSubGraph("sg0");
  auto in0 = sg0.hostFloat64Variable({});
  auto in1 = in0.variable();
  auto out = in0 * in1;

  auto sg1 = m.createSubGraph("sg1");
  auto in2 = in0.variable(sg1);
  auto in3 = in2.variable();
  auto rpt = sg1.repeatAllOut(
      sg0, 3, {}, {{{in3, in1, out}, {in2, in0, in0}}}, {out});

  // out =  in0 * in1
  //     -> in0 * (in0 * in1)
  //     -> in0 ** 3 * in1.
  //
  // dout / din0 = 3 in0 ** 2 * in1.
  auto out1    = out.dstInCaller(rpt);
  auto loss    = out1.reduceSum(Shape{});
  auto in2Grad = m.tensor(Autodiffer(m).backward(loss, {in2})[0]).copy();

  m.setRunnable({sg1});

  setCompiledSlickGraph(m);

  auto init2 = HostTensor::float64(5);
  auto init3 = HostTensor::float64(3);

  cm().setHostValue(in2, init2);
  cm().setHostValue(in3, init3);

  cm().run(sg1);

  cm().getHostValue(in2Grad).assertAllEquivalent(
      init2.pow(2).mul(init3).mul(3));
}

void RepeatTester::testManual0() {

  SlickGraph m;

  auto sg0 = m.createSubGraph("sg0");
  auto in0 = sg0.hostFloat64Variable({});
  auto out = in0.cos().sin();

  auto sg1 = m.createSubGraph("sg1");
  auto in1 = in0.variable(sg1);
  auto rpt = sg1.repeatAllOut(sg0, 2, {}, {{{in1, in0, out}}}, {{out}});

  Autodiffer ad(m);
  auto sg2 = ad.backwardOutOfGraph({out}, {in0}, {in0});

  ad.setGrad(rpt, CalleeIndex(0), sg2);

  auto loss = out.dstInCaller(rpt).reduceSum(Shape{});
  auto dIn1 = ad.backward(loss, {in1})[0];

  m.setRunnable({sg1});
  setCompiledSlickGraph(m);
  m.verifyValid();

  // sin(cos(sin(cos(x)))) has gradient:
  // cos(cos(sin(cos(x)))) * (sin(sin(cos(x)))) * cos(cos(x)) * sin(x).
  auto one      = HostTensor::float64(1);
  auto expected = one.cos().sin().cos().cos() * one.cos().sin().sin() *
                  one.cos().cos() * one.sin();

  cm().setHostValue<double>(in1, {1.0});
  cm().run(sg1);
  cm().getHostValue(dIn1).assertAllClose(
      expected, 1e-5, 1e-5, "vs hand constructed gradient");
}

void RepeatTester::testTowardsNoExit0() {

  SlickGraph m;
  auto sg0 = m.createSubGraph("sg0");
  auto in0 = sg0.hostFloat64Variable({2});
  auto x0  = in0 * in0;
  auto x1  = x0.sin();

  auto sg1 = m.createSubGraph("sg1");
  auto in1 = in0.variable(sg1);
  auto rpt = sg1.repeatAllOut(sg0, 3, {}, {{{in1, in0, x0}}}, {{x1}});

  auto loss = x1.dstInCaller(rpt).reduceSum();
  auto dIn1 = Autodiffer(m).backward(loss, {in1})[0];
  (void)dIn1;

  m.setRunnable({sg1});

  setCompiledSlickGraph(m);

  auto initVal = HostTensor::float64({2}, {1., -0.5});

  cm().setHostValue(in1, initVal);
  cm().run(sg1);

  // out = sin(in^8))
  // dout = cos(in^8).8.in^7.
  cm().getHostValue(dIn1).assertAllClose(initVal.pow(7).mul(8) *
                                             initVal.pow(8).cos(),
                                         1e-6,
                                         1e-6,
                                         "hand crafted gradient");
}

void RepeatTester::testMultiProngOut0() {

  Shape s0{7};
  auto init0 = HostTensor::arangeFloat32(-3, 4, 1);

  auto valueWithRepeat = [this, s0, init0](bool copyPhi) {
    SlickGraph m;

    auto sg0 = m.createSubGraph("sg0");
    auto in0 = sg0.hostFloat32Variable(s0);
    auto x0  = in0 * in0;

    auto phi = copyPhi ? x0.copy() : x0;
    auto x1  = x0.div(x0.constant(2.));

    auto sg1 = m.createSubGraph("sg1");
    auto in1 = in0.variable(sg1);
    auto rpt = sg1.repeat(sg0,
                          3,
                          {},
                          {{{in1, in0, x0}}},
                          {{{in0, IsStackedCopy::Yes},
                            {phi, IsStackedCopy::Yes},
                            {x1, IsStackedCopy::No}}});

    auto loss = phi.dstInCaller(rpt).reduceSum(Shape{}) +
                x1.dstInCaller(rpt).reduceSum(Shape{});

    auto dIn1 = m.tensor(Autodiffer(m).backward(loss, {in1})[0]).copy();
    m.setRunnable({sg1});

    setCompiledSlickGraph(m);
    cm().setHostValue(in1, init0);
    cm().run(sg1);
    return cm().getHostValue(dIn1);
  };

  {
    bool caught{false};
    try {
      valueWithRepeat(false);
    } catch (const poprithms::error::error &e) {
      caught = true;
      std::string em(e.what());
      if (em.find("non-zero derivative") == std::string::npos) {
        std::ostringstream oss;
        oss << "Incorrect error message, " << em
            << " doesn't contain expected sub-string";
        throw poprithms::test::error(oss.str());
      }
    }
    if (!caught) {
      throw poprithms::test::error(
          "Failed to catch error of stacked tensor to loss which is "
          "also carried back");
    }
  }

  auto vRep = valueWithRepeat(true);

  auto vDir = [this, s0, init0]() {
    SlickGraph m;
    auto sg0 = m.createSubGraph("sg0");
    auto in0 = sg0.hostFloat32Variable(s0);
    auto x0  = in0;
    // auto x1 = x0;

    auto loss = sg0.constant(DType::Float32, 0., m.host());
    for (uint64_t i = 0; i < 3; ++i) {
      x0   = x0 * x0;
      loss = loss + x0;
    }

    auto x1 = x0 / x0.constant(2);
    loss    = loss + x1;
    loss    = loss.reduceSum();
    auto d0 = m.tensor(Autodiffer(m).backward(loss, {in0})[0]).copy();
    m.setRunnable({sg0});

    setCompiledSlickGraph(m);
    cm().setHostValue(in0, init0);
    cm().run(sg0);
    return cm().getHostValue(d0);
  }();

  vDir.assertAllEquivalent(vRep);
}

void RepeatTester::testLastMinuteZero() {
  SlickGraph m;
  auto sg0  = m.createSubGraph("sg0");
  auto in0  = sg0.hostFloat64Variable({});
  auto out0 = in0 + in0.constant(1);
  auto out1 = out0 * in0;

  //      out0      out1
  // 0 :  in0+1     (in0+1)*in0
  // 1 :  in0+2     (in0+2)*(in0+1)
  // 2 :  in0+3     (in0+3)*(in0+2)
  //
  // loss = c0*( in0**2  +  in0 + 0) +
  //        c1*( in0**2  + 3in0 + 2) +
  //        c2*( in0**2  + 5in0 + 6).
  //
  //      = (c0 + c1 + c2)in0**2 + (c0  + 3c1 + 5c2)in0 + 2c1 + 6c2.
  //
  // dloss = 2(c0 + c1 + c2)in0 + (c0 + 3c1 + 5c2).
  //

  auto sg1 = m.createSubGraph("sg1");
  auto in1 = in0.variable(sg1);
  auto rpt =
      sg1.repeat(sg0,
                 3,
                 {},
                 {{{in1, in0, out0}}},
                 {{{out1, IsStackedCopy::Yes}, {in0, IsStackedCopy::Yes}}});

  auto out1InMain = out1.dstInCaller(rpt);

  double c0{1};
  double c1{2};
  double c2{3};
  auto loss = (out1InMain *
               out1InMain.constant(HostTensor::float64({3}, {c0, c1, c2})))
                  .reduceSum();

  auto dIn1 = m.tensor(Autodiffer(m).backward(loss, {in1})[0]).copy();

  m.setRunnable({sg1});

  setCompiledSlickGraph(m);

  auto hv = HostTensor::float64(2);
  cm().setHostValue(in1, hv);
  cm().run(sg1);

  auto expectedGrad = hv.mul(2).mul(c0 + c1 + c2).add(c0 + c1 * 3 + c2 * 5);

  cm().getHostValue(dIn1).assertAllEquivalent(expectedGrad);
}

void RepeatTester::testAutodiff0() {

  SlickGraph m;
  auto sg0 = m.createSubGraph("sg0");
  auto in0 = sg0.hostFloat64Variable({});
  auto x0  = in0 + in0.constant(1);
  auto y0  = x0 * x0;

  auto sg1  = m.createSubGraph("sg1");
  auto in1  = in0.variable(sg1);
  auto rpt  = sg1.repeat(sg0,
                        3,
                        {},
                        {{{in1, in0, x0}}},
                        {{{x0, IsStackedCopy::Yes},
                           {in0, IsStackedCopy::Yes},
                           {y0, IsStackedCopy::No}}});
  auto loss = y0.dstInCaller(rpt);

  auto dIn1 = Autodiffer(m).backward(loss, {in1})[0];

  (void)dIn1;

  // y0_final = (x0_0 + 3)^2

  m.setRunnable({sg1});
  setCompiledSlickGraph(m);
  auto hv    = HostTensor::float64(2);
  auto grad0 = hv.add(3).mul(2);
  cm().setHostValue(in1, hv);
  cm().run(sg1);

  cm().getHostValue(dIn1).assertAllEquivalent(grad0);
}

void RepeatTester::testMixedBag0() {

  SlickGraph m;

  //     in0float
  //        |
  //     +--+   in0int---+
  //     |  |      |     |
  //     |  |     cast   |
  //     |  |      |     |
  //     |  +---+--+     +1 (aka addOut)
  //     |      |
  //     |     mul (aka mulOut)
  //   pow(2) (aka powOut)
  //
  auto sg0      = m.createSubGraph("sg0");
  auto in0int   = sg0.hostInt32Variable({});
  auto in0float = sg0.hostFloat64Variable({});
  auto mulOut   = in0float.mul(in0int.to(in0float.dtype())).name("mulOut");
  auto addOut   = in0int + in0int.constant(1).name("addOut");
  auto powOut   = in0float.pow(in0float.constant(2)).name("powOut");

  uint64_t rptCount{3};
  auto sg1      = m.createSubGraph("sg1");
  auto in1float = in0float.variable(sg1);
  auto in1int   = sg1.variable(
      in0int.dtype(), in0int.shape().prepend(rptCount), m.host());

  auto rpt = sg1.repeat(sg0,
                        rptCount,
                        {{{in1int, in0int}}},
                        {{{in1float, in0float, mulOut}}},
                        {{{powOut, IsStackedCopy::No},
                          {addOut, IsStackedCopy::Yes},
                          {in0int, IsStackedCopy::Yes},
                          {in0float, IsStackedCopy::Yes}}});

  auto loss =
      powOut.dstInCaller(rpt) *
      addOut.dstInCaller(rpt).reduceMax().to(powOut.dtype()).name("loss!");

  auto din1float = m.tensor(Autodiffer(m).backward(loss, {in1float})[0])
                       .name("dIn1Float")
                       .copy();

  // easy to see the value of powOut at the end is (floatIn * intIn[0] *
  // intIn[1])**2 Assumint intIn[2] is the largest, loss is therefore
  // floatIn**2 * (intIn[0]**2 * intIn[1]**2 * (intIn[2] + 1))

  m.setRunnable({sg1});

  setCompiledSlickGraph(m);
  cm().setHostValue(in1float, HostTensor::float64(5));
  cm().setHostValue(in1int, HostTensor::int32({3}, {2, 5, 6}));
  cm().run(sg1);

  double expected = 2 * 5 * (2 * 2) * (5 * 5) * (6 + 1);
  cm().getHostValue(din1float).assertAllEquivalent(
      HostTensor::float64(expected));
}

void RepeatTester::testReverseOrder0() {

  auto withOrder = [this](StackedCopyOrder order) {
    SlickGraph m;
    auto sg0  = m.createSubGraph("sg0");
    auto in0  = sg0.hostFloat64Variable({});
    auto in1  = in0.variable();
    auto out0 = in0.pow(in1) - in0.constant(1);

    auto sg1 = m.createSubGraph("sg1");
    auto in2 = in0.variable(sg1);
    auto in3 = in2.variable(Shape{3});
    auto rpt = sg1.repeat(sg0,
                          3,
                          {{{in3, in1}}},
                          {{{in2, in0, out0}}},
                          {{{in0, IsStackedCopy::Yes},
                            {in1, IsStackedCopy::Yes},
                            {out0, IsStackedCopy::No}}},
                          order);

    auto ado = Autodiffer(m).backward({rpt, 2}, {in2});

    m.setRunnable({sg1});
    setCompiledSlickGraph(m);
    cm().setHostValue(in2, HostTensor::float64(2));
    cm().setHostValue(in3, HostTensor::float64({3}, {1, 1, 3}));
    cm().run(sg1);

    // Down: out = ((x^3 - 1)^1 - 1)^1 - 1 = x^3 - 3. dOut = 3x^2.
    // Up  : out = ((x^1 - 1)^1 - 1)^3 - 1 = (x - 2)^3 - 1. dOut = 3(x-2)^2.

    if (order == StackedCopyOrder::Up) {
      cm().getHostValue(ado[0]).assertAllEquivalent(HostTensor::float64(0));
    } else {
      cm().getHostValue(ado[0]).assertAllEquivalent(HostTensor::float64(12));
    }
  };
  withOrder(StackedCopyOrder::Up);

  bool caught{false};
  try {
    withOrder(StackedCopyOrder::Down);
  } catch (const poprithms::error::error &e) {

    // TODO: a task number for backprop through downwards.

    caught        = true;
    std::string w = e.what();
    auto found    = w.find("will require associating an order");
    if (found == std::string::npos) {
      throw poprithms::test::error("Expected the error to explain the plan");
    }
  }

  if (!caught) {
    throw poprithms::test::error(
        "can you backprop with StackedCopyOrder::Down now?");
  }
}

void RepeatTester::testRepeatInCall0() {
  SlickGraph m;

  SubGraph sg0 = m.createSubGraph("sg0");
  auto x0      = sg0.hostFloat64Variable({});
  auto out0    = x0.pow(x0.constant(2));

  SubGraph sg1 = m.createSubGraph("sg1");
  auto x1      = x0.variable(sg1);
  auto rpt     = sg1.repeatAllOut(sg0, 3, {}, {{{x1, x0, out0}}}, {});
  auto out1    = out0.dstInCaller(rpt);
  out1         = out1 + out1.constant(3);

  SubGraph sg2 = m.createSubGraph("sg2");
  auto x2      = x1.variable(sg2);
  auto call2   = sg2.callAllOut(sg1, {{{x2, x1}}});
  auto psi     = out1.dstInCaller(call2).sin();

  // psi = (x2.pow(8) + 3).sin()
  auto dx2 = m.tensor(Autodiffer(m).backward(psi, {x2})[0]).copy();

  m.setRunnable({sg2});

  setCompiledSlickGraph(m);
  auto ht = HostTensor::float64(0.8);
  cm().setHostValue(x2, ht);
  cm().run(sg2);

  auto a = cm().getHostValue(dx2);
  auto b = (ht.pow(8).add(3)).cos().mul(8).mul(ht.pow(7));
  a.assertAllClose(b, 1e-5, 1e-5);
}

void RepeatTester::testSimpleInfer0() {

  const auto rf = ReplicationFactor::create(1);
  SlickGraph m(1, rf);

  auto sg0 = m.createSubGraph("productGraph");
  auto x0  = sg0.variable(DType::Int32, {}, m.rootIpu());
  auto x1  = x0.variable();
  auto out = x0 * x1;

  auto sg1 = m.createSubGraph("main");
  auto x2  = sg1.hostInt32Variable({1, 1});
  auto x3  = x2.hostToIpu(m.rootIpu());
  auto rpt = sg1.repeat(sg0,
                        5,
                        {},
                        {{{x3, x0, x0}, {x3, x1, out}}},
                        {{{out, IsStackedCopy::No}}});

  auto backOnHost = m.tensor({rpt, 0}).ipuToHost(1);

  m.setRunnable({sg1});

  setCompiledSlickGraph(m);

  cm().setHostValue<int>(x2, {2});
  cm().run(sg1);
  cm().getHostValue(backOnHost).assertAllEquivalent(HostTensor::int32(64));
}

void RepeatTester::testDynamicUpdateInRepeat0() {

  uint64_t nTilesPerReplica = 10;
  SlickGraph m(nTilesPerReplica, ReplicationFactor::create(1));
  auto sg = m.createSubGraph("sg");

  auto sliceable0 = sg.variable(DType::Float32, {4}, m.rootIpu());
  auto slice0     = sg.variable(DType::Float32, {2}, m.rootIpu());
  auto offset0    = sg.variable(DType::Unsigned32, {1}, m.rootIpu());
  auto out = sliceable0.dynamicUpdate_(slice0, offset0, Dimensions{0}).copy();

  auto updatedOffset = offset0 + offset0.constant(1);

  auto sg1        = m.createSubGraph("sg1");
  auto sliceable1 = sg1.variable(DType::Float32, {1, 1, 4}, m.host());
  auto slice1     = sg1.variable(DType::Float32, {1, 1, 2}, m.host());
  auto offset1    = sg1.variable(DType::Unsigned32, {1, 1, 1}, m.host());

  auto rpt =
      sg1.repeat(sg,
                 3,
                 {},
                 {{
                     {sliceable1.hostToIpu(m.rootIpu()), sliceable0, out},
                     {slice1.hostToIpu(m.rootIpu()), slice0, slice0},
                     {offset1.hostToIpu(m.rootIpu()), offset0, updatedOffset},
                 }},
                 {{{out, IsStackedCopy::No}}});

  auto backOnHost = m.tensor({rpt, 0}).ipuToHost(1);
  m.setRunnable({sg1});

  setCompiledSlickGraph(m);

  cm().setHostValue<float>(sliceable1, {5, 6, 7, 8});
  cm().setHostValue<float>(slice1, {2, 3});
  cm().setHostValue<uint32_t>(offset1, {0});
  cm().run(sg1.id());

  auto vOut = cm().getHostValue(backOnHost);
  vOut.assertAllEquivalent(HostTensor::float32({4}, {2, 2, 2, 3}));
}

} // namespace testutil
} // namespace compute
} // namespace common
} // namespace poprithms
