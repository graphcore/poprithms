// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <sstream>

#include <poprithms/autodiff/guide/graphinfo.hpp>
#include <poprithms/autodiff/guide/guide.hpp>
#include <poprithms/common/compute/autodiff/autodiffer.hpp>
#include <poprithms/common/compute/autodiff/automaticmutator.hpp>
#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/ops/withoutcallees.hpp>
#include <poprithms/common/compute/slickgraph.hpp>
#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>

namespace {
using namespace poprithms::common::compute;
namespace difftest {

//
// A custom op for testing, with aliases between inputs and outputs.
//
// It has 3 inputs, and 3 outputs, described below:
//
//  0   a ----+        +----->  a * b
//            |        |
//  1   b --- +--CustomOp --+----->  b * i.castTo(b)
//            |        |
//  2   i ----+        +----->  (i + 2) % 5
//

class CustomOp final : public WithoutCalleesTensorCentric {
public:
  CustomOp(const State &s) : WithoutCalleesTensorCentric(s) {}

  void resetRootRef(OutIndex, const TensorId &) { invalid(); }

  UpOp cloneWithState(const State &s) const final {
    return std::unique_ptr<CustomOp>(new CustomOp(s));
  }

  bool isValueDependent(InIndex, OutIndex) const final { return true; }

  TensorId rootRef(OutIndex o) const final { return outTensorId(o); }

  void computeDerivedVerifyValid() const final {}

  void assertValid() const {}

  void computeDerivedRemoveInputs(const ContiguousInIndexSubset &) final {}

  void computeDerivedRemoveOutputs(const ContiguousOutIndexSubset &) final {}

  std::string typeString() const final { return "CustomOp"; }

  bool aliases(InIndex, OutIndex) const final { return false; }

  bool modifies(InIndex) const final { return false; }

  void growAliasMapper(MemoryAliasMapper &b) const final {
    createVariables(b);
  }

  bool isInitializingOp() const final { return false; }

  CodeLocation codeLocation() const { return locationByUnanimity(); }

  virtual bool gradientPropagates(OutIndex o, InIndex i) const {
    return (o == 0 && i == 0) || (o == 0 && i == 1) || (o == 1 && i == 1);
  }

  // See the diagram above.
  void compute(const HostTensors &ins, const HostTensors &outs) const final {
    outs[0].update_(ins[0] * ins[1]);
    outs[1].update_(ins[1] * ins[2].to(ins[1].dtype()));
    outs[2].update_((ins[2].increment(2)).mod(5));
  }

  // Return the gradients of the inputs. See the diagram above.
  OptionalTensors bprop(const GradOpIns &gIn) const final {

    auto dOut0 = gIn.gradOfOutput(0);
    auto dOut1 = gIn.gradOfOutput(1);
    auto a     = gIn.input(0);
    auto b     = gIn.input(1);
    auto i     = gIn.input(2);

    auto da = (dOut0 * b).reduceSum(inShape(0));
    auto db = (dOut0 * a).reduceSum(inShape(1)) +
              (dOut1 * i.to(a.dtype())).reduceSum(inShape(1));

    return {da, db, {}};
  }

  std::vector<InIndex> autodiffRequiredIns() const final { return {0, 1, 2}; }

  std::vector<OutIndex> autodiffRequiredOuts() const final { return {}; }

  HostTensors initializeOut(const HostTensors &) const final {
    return zeroOuts();
  }

  void initializeSimOut(SimTensorMap &htm) const final {
    initializeReplicatedSimOut(htm);
  }

  void runSim(ISimState &ss) const final {
    runReplicatedSim(ss.simTensorMap());
  }

  bool computeTypeSpecificEqualTo(const Op &) const { return true; }
};

// Extension of the SliceGraph which has a method to insert a custom op.
class TestGraph : public SlickGraph {
public:
  TestGraph() : SlickGraph() {}
  Tensors customOp(Tensor a, Tensor b, Tensor i) {
    const auto oShape = a.shape().numpyBinary(b.shape());
    auto opId         = createComputeOp<difftest::CustomOp>(
        {a.id(), b.id(), i.id()},
        subGraphId(a.id()),
        TensorInfos({tensorInfo(a).withShape(oShape),
                             tensorInfo(b).withShape(oShape),
                             tensorInfo(i)}));

    Tensors ts;
    for (uint64_t o = 0; o < nOutTensors(opId); ++o) {
      ts.push_back(tensor({opId, o}));
    }
    return ts;
  }
};

} // namespace difftest
} // namespace

int main() {
  difftest::TestGraph m;
  auto sg0 = m.createSubGraph("sg0");
  auto a0  = sg0.variable(DType::Float32, {3, 3}, m.host());
  auto a1  = a0.sin();
  auto b0  = sg0.variable(DType::Float32, {2, 3, 3}, m.host());
  auto b1  = b0.cos();
  auto i   = sg0.variable(DType::Int16, {}, m.host());
  auto ipi = i + i;

  auto outs = m.customOp(a1, b1, ipi);
  auto f0   = outs[0];
  auto f1   = outs[1];
  auto f2   = outs[2];

  //
  //                      i
  //        a0    b0      |
  //        |     |    +--+--+
  //        v     v    |     |
  //        |     |    +-+ +-+
  //      (sin) (cos)    | |
  //        |     |     (add)
  //       a1    b1       |
  //        |     |       |
  //        +-----+-------+
  //              |
  //            (foo)
  //              |
  //        +-----+-----+
  //        |     |     |
  //       f0     f1    f2
  //
  //        ^     ^
  //       gIn   gIn

  Autodiffer<difftest::TestGraph, AutomaticMutator> ad(m);
  auto test = [&m, &ad](const TensorIds &gradsProvidedFor,
                        const TensorIds &checkpoints,
                        const TensorIds &targets,
                        const std::set<OpId> &expectedOpsToRerun,
                        const std::set<TensorId> &expectedNonGradsWithGrads,
                        bool expectError = false) {
    auto objective = poprithms::autodiff::guide::Objective::outOfGraph(
        gradsProvidedFor, checkpoints, targets);
    if (expectError) {
      bool caught{false};
      try {

        poprithms::autodiff::guide::Guide d(
            objective, GuideGraphInfo(m, ad.gradInfos()));
        caught = true;
      } catch (const poprithms::error::error &) {
        caught = true;
      }
      if (!caught) {
        std::ostringstream oss;
        oss << "Expected to catch an error for the Objective=" << objective;
        throw poprithms::test::error(oss.str());
      }
    } else {

      poprithms::autodiff::guide::Guide d(objective,
                                          GuideGraphInfo(m, ad.gradInfos()));
      const auto obseredOpsToRerun    = d.opsToRerun();
      const auto obseredOpsToRerunSet = std::set<OpId>(
          obseredOpsToRerun.cbegin(), obseredOpsToRerun.cend());
      if (obseredOpsToRerunSet != expectedOpsToRerun) {
        std::ostringstream oss;
        oss << "Not the expected set of ops to rerun. ";
        oss << "\nobserved=";
        poprithms::util::append(oss, obseredOpsToRerun);
        oss << "\nexpected=";
        poprithms::util::append(
            oss,
            OpIds(expectedOpsToRerun.cbegin(), expectedOpsToRerun.cend()));
        oss << "This for objective=" << objective << ", \nand machine ";
        m.append(oss);

        throw poprithms::test::error(oss.str());
      }

      const auto ngwg = d.nonGradsWithGrads();
      if (expectedNonGradsWithGrads != ngwg) {
        std::ostringstream oss;
        oss << "Not the expected set of tensors with gradients. ";
        oss << "\nobserved=";
        poprithms::util::append(oss, TensorIds(ngwg.cbegin(), ngwg.cend()));
        oss << "\nexpected=";
        poprithms::util::append(oss,
                                TensorIds(expectedNonGradsWithGrads.cbegin(),
                                          expectedNonGradsWithGrads.cend()));
        oss << "This for objective=" << objective << ", \nand machine ";
        m.append(oss);

        throw poprithms::test::error(oss.str());
      }
    }
  };

  test({f0, f1},                // grads provided for
       {a0, b0, ipi},           // checkpoints
       {a0, b0},                // targets
       {a1.opId(), b1.opId()},  // expects ops to rerun
       {a0, a1, b0, b1, f0, f1} // expected non grads with grads
  );

  test({f0, f1},                // grads provided for
       {a0, b0, ipi, a1, b1},   // checkpoints
       {a0, b0},                // targets
       {},                      // expects ops to rerun
       {a0, a1, b0, b1, f0, f1} // expected non grads with grads
  );

  test({f0, f1},                           // grads provided for
       {a0, b0, i},                        // checkpoints
       {a0, b0},                           // targets
       {a1.opId(), b1.opId(), ipi.opId()}, // expects ops to rerun
       {a0, a1, b0, b1, f0, f1}            // expected non grads with grads

  );

  test({f0},                               // grads provided for
       {a0, b0, i},                        // checkpoints
       {a0},                               // targets
       {a1.opId(), b1.opId(), ipi.opId()}, // expects ops to rerun
       {a0, a1, f0, f1}                    // expected non grads with grads
  );

  test({f0, f1},                           // grads provided for
       {a0, b0, i},                        // checkpoints
       {b0},                               // targets
       {a1.opId(), b1.opId(), ipi.opId()}, // expects ops to rerun
       {b0, b1, f0, f1}                    // expected non grads with grads
  );

  test({f0},                    // grads provided for
       {a0, b1, i},             // checkpoints
       {a0},                    // targets
       {a1.opId(), ipi.opId()}, // expects ops to rerun
       {a0, a1, f0, f1}         // expected non grads with grads
  );

  // f2 is an integer:
  test({f0, f1, f2}, // grads provided for
       {a0, b1, i},  // checkpoints
       {a0},         // targets
       {},
       {},
       true);

  return 0;
}
