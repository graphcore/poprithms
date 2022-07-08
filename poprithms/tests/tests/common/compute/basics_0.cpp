// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>

#include <poprithms/common/compute/initialvalues.hpp>
#include <poprithms/common/compute/ops/init.hpp>
#include <poprithms/common/compute/ops/reffrom.hpp>
#include <poprithms/common/compute/simtensormap.hpp>
#include <poprithms/common/compute/slickgraph.hpp>
#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/util/interval.hpp>

namespace {
using namespace poprithms::common::compute;
void testInitialValues0() {

  InitialValues inVals(2);
  inVals.setValue(OutIndex(0), 3, HostTensor::float32(17));

  auto inVals2 = inVals;

  if (inVals2 != inVals) {
    throw poprithms::test::error("Comparison of copied InitialValues failed");
  }

  InitialValues inVals3(2);
  inVals3.setValue(OutIndex(0), 3, HostTensor::float32(17));

  if (inVals3 != inVals) {
    throw poprithms::test::error(
        "Comparison of numerically equivalent InitialValues failed");
  }

  InitialValues inVals4(2);
  inVals4.setValue(OutIndex(0), 3, HostTensor::float32(17.001));

  if (inVals4 == inVals) {
    throw poprithms::test::error(
        "Comparison of numerically different InitialValues failed");
  }

  InitialValues inVals5(2);
  inVals5.setValue(OutIndex(0), 3, inVals.getInitialValues(0).at(3).copy());
  if (inVals5 != inVals) {
    throw poprithms::test::error(
        "Comparison of numerically equivalent InitialValues failed, value "
        "obtained my introspection");
  }
}

/**
 * An op which makes a copy of its input, with certain functionality not
 * implemented yet as not required for testing.
 * */
class TestCopy : public WithoutCallees {

public:
  std::string typeString() const final { return "testop"; }
  CodeLocation codeLocation() const final { unimplemented("codeLocation"); }
  bool computeTypeSpecificEqualTo(const compute::Op &) const final {
    return true;
  }

  void resetRootRef(OutIndex, const TensorId &) final { invalid(); }

  bool isInitializingOp() const final { return false; }

  void runSim(ISimState &) const final { unimplemented("runSim"); }

  void initializeSimOut(SimTensorMap &) const final {
    unimplemented("initializeSimOut");
  }

  HostTensors initializeOut(const HostTensors &) const final {
    unimplemented("initializeOut");
  }

  void computeDerivedRemoveInputs(const ContiguousInIndexSubset &) final {}
  void computeDerivedRemoveOutputs(const ContiguousOutIndexSubset &) final {}

  void computeDerivedVerifyValid() const final {}

  bool aliases(InIndex, OutIndex) const final { return false; }

  virtual bool modifies(InIndex) const final { return false; }

  bool gradientPropagates(OutIndex, InIndex) const final { return true; }
  TensorId rootRef(OutIndex o) const final { return outTensorId(o); }

  std::unique_ptr<compute::Op> cloneWithState(const State &s) const final {
    return std::make_unique<TestCopy>(s);
  }

  void growAliasMapper(MemoryAliasMapper &mam) const final {
    createVariables(mam);
  }

  ~TestCopy() override = default;
  TestCopy(const poprithms::common::compute::Op::State &s)
      : WithoutCallees(s) {}

  void compute(const HostTensors &ins, const HostTensors &outs) const final {
    ins.at(0).update_(outs.at(0));
  }

  std::vector<InIndex> autodiffRequiredIns() const final { return {}; }

  std::vector<OutIndex> autodiffRequiredOuts() const final { return {}; }

  OptionalTensorIds backpropagate(Graph &,
                                  const GradOpInIds &gIn) const final {
    return {gIn.gradOfOutput(0)};
  }
};

/**
 * Minimal completion of compute::TestGraph to make it non-abstract.
 * */
class TestGraph : public SlickGraph {

private:
public:
  TestGraph() = default;
  TestGraph(uint64_t nTilesPerReplica, ReplicationFactor rf)
      : SlickGraph(nTilesPerReplica, rf) {}

  // Insert a variable
  TensorId var(SubGraphId sgId) {
    auto opId =
        createComputeOp<VarInit>({}, sgId, {TensorInfo({}, 0, DType::Int32)});
    return {opId, 0};
  }

  // Insert a copy
  TensorId copy(const TensorId &tId) {
    auto opId = createComputeOp<TestCopy>(
        {tId}, subGraphId(tId), {TensorInfo({}, 0, DType::Int32)});
    return {opId, 0};
  }

  // Insert a cross-graph reference.
  TensorId refFrom(const TensorId &root, SubGraphId sgId) {
    return tRefFrom<RefFrom>(root, sgId);
  }
};

void testRefAcrossSubGraphs0() {
  TestGraph g;

  auto sg0 = g.createSubGraphId("sg0");
  auto sg1 = g.createSubGraphId("sg1");
  auto sg2 = g.createSubGraphId("sg2");

  const auto in0 = g.copy(g.var(sg0));
  const auto in2 = g.copy(g.var(sg2));

  const auto ref0to1 = g.refFrom(in0, sg1);
  const auto ref2to0 = g.refFrom(in2, sg0);

  const auto base = g.var(sg0);

  if (g.hasDerivedRefs(ref0to1)) {
    throw poprithms::test::error("ref0to1 does not have derived references "
                                 "(it is not a root reference)");
  }

  if (g.isRootRef(ref0to1)) {
    throw poprithms::test::error("ref0to1 is not a root reference");
  }

  if (!g.hasDerivedRefs(in0)) {

    g.append(std::cout);
    std::cout << std::endl;

    throw poprithms::test::error(
        "in0 does have a derived reference (ref0to1)");
  }

  if (!g.isRootRef(base)) {
    throw poprithms::test::error(
        "base is a root reference (singleton equivalence class)");
  }

  if (!g.refsExcludingSelf(base).empty()) {
    throw poprithms::test::error(
        "base is the only element in the equilvalence class, it should not "
        "have refs");
  }

  if (g.refsExcludingSelf(ref2to0).size() != 1) {
    throw poprithms::test::error("ref2to0 should have 1 reference: in2");
  }
  if (g.refsExcludingSelf(in2).size() != 1) {
    throw poprithms::test::error("in2 should have 1 reference: ref2to0");
  }
}

void testRefAcrossSubGraphs1() {

  TestGraph g;
  auto sg0 = g.createSubGraphId("sg0");
  auto sg1 = g.createSubGraphId("sg1");
  auto sg2 = g.createSubGraphId("sg2");

  auto in0 = g.var(sg0);
  auto x1  = g.refFrom(in0, sg1);
  auto x2  = g.refFrom(x1, sg2);

  // These should all have no effect, no new ops as refs already made (or same
  // sub-graph).
  g.refFrom(in0, sg0);
  g.refFrom(x2, sg0);
  g.refFrom(x2, sg1);
  g.refFrom(x1, sg0);

  if (g.nOps() != 3) {
    throw poprithms::test::error(
        "There should only be 3 ops in the graph, as the final 4 refFrom "
        "calls are all create references to tensors which already exist (or "
        "are in the same sub-graph)");
  }

  if (g.rootRef(x2) != in0) {
    throw poprithms::test::error(
        "The root reference of x2 is in0 (in0 is the canonical "
        "representative of the group)");
  }
}

void testVirtualGraph0() {

  TestGraph tg(100, ReplicationFactor::create(2));
  auto subDevs = tg.partition(tg.rootIpu(), 4);
  for (uint64_t p = 0; p < 4; ++p) {
    auto s = subDevs.at(p);
    if (tg.ipu(s).tiles().interval(0) !=
        poprithms::util::Interval(p * 25, (p + 1) * 25)) {
      throw poprithms::test::error(
          "Incorrect partitioning of tiles, expected [0,25), [25, 50), "
          "[50,75), [75, 100)");
    }
  }
}

void testBadValOuts() {
  TestGraph tg(100, ReplicationFactor::create(2));

  auto sg0     = tg.createSubGraphId("sg0");
  auto in0     = tg.var(sg0);
  auto badVals = tg.computeOp(in0.opId()).badValOuts();
  if (badVals.size() == 0 || badVals.at(0).nelms() == 0) {
    throw poprithms::test::error(
        "Expected one op with a tensor with 1 element");
  }
  for (const auto &t : badVals) {
    if (!t.allNonZero()) {
      throw poprithms::test::error(
          "All values should be non-zero in initialized values");
    }
  }
}

void testCastsAndGets() {
  TestGraph tg;
  auto sg0     = tg.createSubGraphId("sg0");
  auto in0     = tg.var(sg0);
  auto nonRefs = tg.opIds<VarInit>(sg0);
  auto refs    = tg.opIds<RefFrom>(sg0);
  if (nonRefs.size() != 1 || refs.size() != 0) {
    throw poprithms::test::error(
        "Failed in check of method to get ops of a specific type");
  }

  bool caught{false};
  try {
    tg.castOrThrow<RefFrom>(in0.opId());
  } catch (const poprithms::error::error &) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error(
        "Failed to catch error of invalid cast (method should throw if "
        "dynamic cast fails)");
  }

  auto derRefs = tg.derivedRefs();
  if (derRefs.size() != 0) {
    throw poprithms::test::error("There are no derived refs in the graph, "
                                 "just the one var (output == root)");
  }

  auto sg1 = tg.createSubGraphId("sg1");
  auto in1 = tg.refFrom(in0, sg1);
  (void)in1;
  if (tg.derivedRefs().size() != 1) {
    throw poprithms::test::error("Now there is 1 derived ref in the graph.");
  }

  auto mam = MemoryAliasMapper(tg, {in1});
  if (mam.graph().nTensors() != 2) {
    throw poprithms::test::error(
        "Expected 2 tensors in the memory alias graph: the tensors in "
        "sub-graph 1 and then tensor in sub-graph 0 (from which it is "
        "derived)");
  }

  if (MemoryAliasMapper(tg, {in0}).graph().nTensors() != 1) {
    throw poprithms::test::error(
        "Expected just 1 tensor in this case. The MemoryAliasMapper where "
        "the target is just 1 variable initialization should never contain "
        "more than 1");
  }
}

void testSetRunnable() {
  TestGraph tg;

  auto sg0 = tg.createSubGraphId("sg0");
  auto sg1 = tg.createSubGraphId("sg1");
  auto in0 = tg.var(sg0);
  (void)in0;

  tg.setRunnable({sg0, sg0});

  // Fine, as same as before.
  tg.setRunnable({sg0});

  {
    auto ctg = tg;
    if (ctg.runnable() != SubGraphIds({sg0})) {
      throw poprithms::test::error(
          "Copy of test graph does not have same runnable sub-graphs");
    }
  }

  bool caught{false};
  try {
    tg.setRunnable({sg0, sg1});
  } catch (const poprithms::error::error &) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error(
        "Failed to catch error of setting runnable sub-graphs twice (with "
        "different sub-graphs)");
  }
}

void testIpuCreation0() {

  TestGraph tg(32, ReplicationFactor::create(1));
  if (tg.nDevices() != 2) {
    throw poprithms::test::error("Expected 2 devices to be created during "
                                 "graph construction - host and root ipu");
  }

  auto foo = tg.ipu(tg.rootIpu(), 10, 20);
  auto bar = tg.ipu(tg.rootIpu(), 10, 20);
  if (foo != bar || foo != DeviceId(2)) {
    throw poprithms::test::error("Expected the second ipu with the tiles "
                                 "[10,20) to have the same id as the first");
  }

  if (tg.nonRootIpuDevices().size() != 1 ||
      tg.nonRootIpuDevices().at(0) != foo) {
    throw poprithms::test::error("Expected just the 1 non-root ipu");
  }
}

void testSimTensorMap() {
  SimTensorMap m;
  m.insertCounter(OpId(5), 7);
  m.incrementCounter(OpId(5));
  m.push_back({HostTensors(5, HostTensor::int16(6))});
  m.push_back({HostTensors(4, HostTensor::int16(3))});

  auto m2 = m.clone();
  if (m2->getCounterState(OpId(5)) != 1) {
    throw poprithms::test::error(
        "Cloned SimTensorMap has incorrect counter state");
  }

  if (m2->getTensors({{1, 0}}, 3).back().getInt16(0) != 3) {
    throw poprithms::test::error(
        "Cloned SimTensorMap has incorrect tensor value");
  }
}

} // namespace

int main() {

  testRefAcrossSubGraphs0();
  testRefAcrossSubGraphs1();
  testInitialValues0();
  testVirtualGraph0();
  testIpuCreation0();
  testSimTensorMap();
  testBadValOuts();
  testSetRunnable();
  testCastsAndGets();
  return 0;
}
