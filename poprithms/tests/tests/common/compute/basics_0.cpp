// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/initialvalues.hpp>
#include <poprithms/common/compute/simtensormap.hpp>
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
 * Minimal completion of compute::Op to make it non-abstract.
 * */
class TestOp : public Op {

public:
  ~TestOp() override = default;

  TestOp(const poprithms::common::compute::Op::State &s) : Op(s) {}

  SubGraphId callee(CalleeIndex) const final { unimplemented("callee"); }
  std::string typeString() const final { return "testop"; }
  CodeLocation codeLocation() const final { unimplemented("codeLocation"); }
  bool computeTypeSpecificEqualTo(const Op &) const final {
    unimplemented("computeTypeSpecificEqualTo");
  }

  SubGraphIds callees() const final { unimplemented("callees"); }

  InIndex inIndex(const CalleeTensorId &) const final {
    unimplemented("inIndex");
  }
  OutIndex outIndex(const CalleeTensorId &) const final {
    unimplemented("outIndex");
  }
  uint64_t nCallees() const final { unimplemented("nCallees"); }

  bool isInitializingOp() const final { unimplemented("isInitializingOp"); }

  void runSim(SimTensorMap &) const final { unimplemented("runSim"); }

  void initializeSimOut(SimTensorMap &) const final {
    unimplemented("initializeSimOut");
  }

  HostTensors initializeOut(const HostTensors &) const final {
    unimplemented("initializeOut");
  }
};

class TestRefFrom final : public TestOp {
public:
  TensorId rr;

  TensorId rootRef(OutIndex) const final { return rr; }

  std::unique_ptr<poprithms::common::multiout::Op>
  cloneMultioutOp() const final {
    return std::make_unique<TestRefFrom>(*this);
  }

  ~TestRefFrom() override = default;
  TestRefFrom(const poprithms::common::compute::Op::State &s,
              const TensorId rootRef)
      : TestOp(s), rr(rootRef) {}
};

class TestNonRefFrom : public TestOp {

public:
  TensorId rootRef(OutIndex o) const final { return outTensorId(o); }

  std::unique_ptr<poprithms::common::multiout::Op>
  cloneMultioutOp() const final {
    return std::make_unique<TestNonRefFrom>(*this);
  }

  ~TestNonRefFrom() override = default;
  TestNonRefFrom(const poprithms::common::compute::Op::State &s)
      : TestOp(s) {}
};

/**
 * Minimal completion of compute::TestGraph to make it non-abstract.
 * */
class TestGraph : public Graph {

private:
  [[noreturn]] void noImpl() const {
    throw poprithms::test::error("not implemented");
  }

public:
  TestGraph() = default;
  TestGraph(uint64_t nTilesPerReplica, ReplicationFactor rf)
      : Graph(nTilesPerReplica, rf) {}

  void multiOutTypeSpecificRemoveOutputs(OpId,
                                         const ContiguousOutIndexSubset &,
                                         const OptionalTensorIds &) final {
    unimplemented("multiOutTypeSpecificRemoveOutputs");
  }
  void
  multiOutTypeSpecificRemoveInputs(OpId,
                                   const ContiguousInIndexSubset &) final {
    unimplemented("multiOutTypeSpecificRemoveInputs");
  }
  bool multiOutTypeSpecificEqualTo(
      const poprithms::common::multiout::Graph &) const final {
    unimplemented("multiOutTypeSpecificEqualTo");
  }
  OpId insertBinBoundary(SubGraphId) final {
    unimplemented("insertBinBoundary");
  }
  std::map<OpId, OpIds>
  schedulableDerivedSpecificConstraints(const OpIds &) const final {
    unimplemented("schedulableDerivedSpecificConstraints");
  }
  void verifyComputeDerivedOpValid(OpId) const final {
    unimplemented("verifyComputeDerivedOpValid");
  }
  void verifyComputeDerivedGraphValid() const final {
    unimplemented("verifyComputeDerivedGraphValid");
  }

  // Insert a variable
  TensorId var(SubGraphId sgId) {
    auto opId = createComputeOp<TestNonRefFrom>(
        {}, sgId, {TensorInfo({}, 0, DType::Int32)});
    return {opId, 0};
  }

  // Insert a relu
  TensorId relu(const TensorId &tId) {
    auto opId = createComputeOp<TestNonRefFrom>(
        {tId}, subGraphId(tId), {TensorInfo({}, 0, DType::Int32)});
    return {opId, 0};
  }

  // Insert a cross-graph reference.
  TensorId refFrom(const TensorId &root, SubGraphId sgId) {
    return tRefFrom<TestRefFrom>(root, sgId);
  }
};

void testRefAcrossSubGraphs0() {
  TestGraph g;

  auto sg0 = g.createSubGraphId("sg0");
  auto sg1 = g.createSubGraphId("sg1");
  auto sg2 = g.createSubGraphId("sg2");

  const auto in0 = g.relu(g.var(sg0));
  const auto in2 = g.relu(g.var(sg2));

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
  return 0;
}
