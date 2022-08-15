// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <iostream>
#include <sstream>

#include <poprithms/common/compute/autodiff/autodiffer.hpp>
#include <poprithms/common/compute/simexecutable.hpp>
#include <poprithms/common/compute/slickgraph.hpp>

namespace {
using namespace poprithms::common::compute;
using Autodiffer = Autodiffer<SlickGraph>;

/**
 * The example at
 * https://pytorch.org/tutorials/beginner/examples_autograd/polynomial_custom_function.html
 * */
void legendrePolynomial3() {

  class LegendrePolynomial3 : public AutogradFunction {
  public:
    // The Legendre polynomial.
    static Tensor l3(Tensor x) {
      return (x.pow(3).mul(5) - x.mul(3)).mul(0.5);
    };

    LegendrePolynomial3(Autodiffer &ad) : AutogradFunction(ad) {}

  private:
    Tensors fwd(const Tensors &ins) final {
      // output 0: is an output because the loss computation requires it.
      // output 1: is an output because it is needed in the backwards pass.
      return {ins[0], l3(ins[0])};
    }

    OptionalTensors bwd(const Tensors &fwd_output,
                        const OptionalTensors &grad_output) final {

      // The gradient of output #1 of fwd:
      auto gradIn = grad_output[1].value();

      // The checkpoint tensor:
      auto fwdIn = fwd_output[0];

      auto correct = gradIn.mul(1.5).mul(fwdIn.pow(2).mul(5).sub(1));
      return {correct.mul(2)};
    }

    bool fwdOutGradUsedInBackwards(OutIndex o) const final { return o == 1; }
  };

  SlickGraph g;
  auto sg0 = g.createSubGraph("sg0");
  auto in0 = sg0.hostFloat32Variable({5});
  Autodiffer ad(g);

  // Using custom gradient:
  LegendrePolynomial3 lp3(ad);
  auto outs = lp3({in0}, "lp3");
  auto out0 = outs[1].reduceSum(Shape{}).name("loss");
  auto d0   = ad.backward(out0, {in0})[0];

  // Using standard gradient:
  auto out1 = LegendrePolynomial3::l3(in0).reduceSum(Shape{});
  auto d1   = Autodiffer(g).backward(out1, {in0})[0];

  g.setRunnable({sg0});

  SimExecutable se(g);
  se.setHostValue(in0, HostTensor::uniformFloat32(-1, 1, {5}, 1011));
  se.run(sg0);

  se.getHostValue(d0).assertAllClose(se.getHostValue(d1).mul(2), 1e-5, 1e-5);
}

void customGrad0() {

  /**
   * Custom gradient method.
   * */
  class BadCalculus : public poprithms::common::compute::AutogradFunction {

  public:
    BadCalculus(Autodiffer &ad) : AutogradFunction(ad) {}

  private:
    /**
     * out = sin(in).
     * */
    Tensors fwd(const Tensors &ins) final {
      auto a = ins[0].sin();
      return {a};
    }

    /**
     * dIn = sin(out)*dOut. It should be cos(in)*dOut.
     * */
    virtual OptionalTensors bwd(const Tensors &outs,
                                const OptionalTensors &outGrads) final {

      auto foo = outGrads[0].value() * outs[0].sin();
      return {foo};
    }

    /**
     * Which tensors in #outGrads of bwd are used in bwds? All (one) of them.
     * */
    bool fwdOutGradUsedInBackwards(OutIndex) const final { return true; }
  };

  // Perform numerical test.
  SlickGraph g;
  Autodiffer ad(g);
  auto sg0 = g.createSubGraph("sg0");
  auto in0 = sg0.hostFloat32Variable({2, 2});
  auto in1 = sg0.hostFloat32Variable({1, 2});
  auto in2 = sg0.hostFloat32Variable({2, 2});
  BadCalculus bc(ad);

  auto loss =
      (bc({in0}, "bc0")[0] + bc({in1}, "bc1")[0] + bc({in2}, "bc2")[0])
          .reduceSum(Shape{});

  auto dIn0 = ad.backward(loss, {in0, in1, in2})[0];
  g.setRunnable({sg0});
  SimExecutable se(g);
  const auto h0 = HostTensor::float32({2, 2}, {1, 2, 3, 4});
  se.setHostValue(in0, h0);
  se.run(sg0);
  se.getHostValue(dIn0).assertAllClose(h0.sin().sin(), 1e-6, 1e-6);
};

void customGrad1() {

  class Fou : public AutogradFunction {
  public:
    Fou(Autodiffer &ad) : AutogradFunction(ad) {}

  private:
    Tensors fwd(const Tensors &ins) final {
      auto out0 = ins[0];
      auto out1 = ins[0] * ins[1];
      auto out2 = ins[1].sin();
      return {out0, out1, out2};
    }

    OptionalTensors bwd(const Tensors &fwd_output,
                        const OptionalTensors &grad_output) final {
      auto g0    = grad_output[0].value();
      auto g2    = grad_output[2].value();
      auto grad1 = fwd_output[0] * fwd_output[1] + g0 * g2;
      return {{}, grad1, {}};
    }

    bool fwdOutGradUsedInBackwards(OutIndex) const final { return true; }
  };

  // out0 = in0
  // out1 = in0 * in1
  // out2 = in1.sin
  //
  // g0 = 1
  // g2 = 1
  //
  // grad2 = out0 * out1  +  g0 * g2
  //       = in0 * in0 * in1 + 1
  //       = 81.

  SlickGraph g0;
  auto sg0 = g0.createSubGraph("sg0");
  auto in0 = sg0.hostFloat32Variable({});
  auto in1 = sg0.hostFloat32Variable({});
  auto in2 = sg0.hostFloat32Variable({});

  Autodiffer ad(g0);
  auto outs  = Fou(ad)({in0, in1, in2});
  auto loss  = (outs[0] + outs[1] + outs[2]).reduceSum(Shape{});
  auto grads = ad.backward(loss, {in0, in1, in2});

  g0.setRunnable({sg0});

  SimExecutable se(g0);
  se.setHostValue(in0, HostTensor::float32(4));
  se.setHostValue(in1, HostTensor::float32(5));
  se.setHostValue(in2, HostTensor::float32(6));

  se.run(sg0);

  // See the hand calculation above.
  se.getHostValue(grads[0]).assertAllEquivalent(HostTensor::float32(0));
  se.getHostValue(grads[1]).assertAllEquivalent(HostTensor::float32(81));
  se.getHostValue(grads[2]).assertAllEquivalent(HostTensor::float32(0));
}

void customGrad2() {

  class Jumper : public AutogradFunction {
  public:
    Jumper(Autodiffer &ad) : AutogradFunction(ad) {}

  private:
    Tensors fwd(const Tensors &ins) final {
      auto out = ins[0].to(DType::Int32).to(DType::Float32);
      return {ins[0], out};
    }

    OptionalTensors bwd(const Tensors &ins,
                        const OptionalTensors &grad_output) final {
      return {ins[0] + ins[1] + grad_output.at(1).value()};
    }
  };

  SlickGraph g;
  auto sg0 = g.createSubGraph("sg0");
  auto in0 = sg0.hostFloat32Variable({});
  Autodiffer ad(g);
  Jumper j(ad);
  auto outs = j({in0}, "jump0");

  // Loss path 0
  auto loss0 = outs[1].modulo(1.0);
  auto dIn0  = ad.backward(loss0, {in0})[0];

  // Loss path 1
  auto loss1 = outs[1].pow(2);
  auto dIn1  = Autodiffer(g).backward(loss1, {in0})[0];

  g.setRunnable({sg0});
  SimExecutable se(g);
  se.setHostValue(in0, HostTensor::float32(13.5));
  se.run(sg0);
  se.getHostValue(loss0).assertAllEquivalent(HostTensor::float32(0));
  se.getHostValue(dIn0).assertAllEquivalent(HostTensor::float32(0));
  se.getHostValue(dIn1).assertAllEquivalent(HostTensor::float32(0));
}

} // namespace

int main() {
  legendrePolynomial3();
  customGrad0();
  customGrad1();
  customGrad2();
  return 0;
}
