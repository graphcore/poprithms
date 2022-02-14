// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <sstream>
#include <vector>

#include <testutil/common/schedulable/graph.hpp>

#include <poprithms/common/schedulable/graph.hpp>
#include <poprithms/common/schedulable/op.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms::common::schedulable_test;

template <typename t>
std::ostream &operator<<(std::ostream &ost, const std::vector<t> &opids) {
  poprithms::util::append(ost, opids);
  return ost;
}

void confirmSame(const Graph &g,
                 const OpIds &observed,
                 const OpIds &expected) {
  if (observed != expected) {
    std::ostringstream oss;
    oss << "Failure in confirming that the schedule " << expected
        << " is created for Graph " << g << ". "
        << "The observed schedule was " << observed << ".";
    throw poprithms::test::error(oss.str());
  }
}

// Basic constraints with OpIds.
void basic0() {

  Graph g;
  const auto gId = g.createSubGraphId("g0");
  const auto in0 = g.insert({}, 1, gId, "input0");
  const auto in1 = g.insert({}, 1, gId, "input1");
  const auto add =
      g.insert({TensorId{in0, 0}, TensorId{in1, 0}}, 1, gId, "add");
  const auto mul =
      g.insert({TensorId{in0, 0}, TensorId{in1, 0}}, 1, gId, "mul");

  {
    Graph g0 = g;
    g0.constraint(in0, in1);
    g0.constraint(add, mul);
    confirmSame(g0, g0.vanillaSchedule(), {in0, in1, add, mul});
  }

  {
    Graph g0 = g;
    g0.constraint(in1, in0);
    g0.constraint(mul, add);
    confirmSame(g0, g0.vanillaSchedule(), {in1, in0, mul, add});
  }

  // using the variadic template
  {
    Graph g0 = g;
    g0.constraint(in1, in0, add, mul);
    confirmSame(g0, g0.vanillaSchedule(), {in1, in0, add, mul});
  }
}

// Sub-graph schedules.
void basic1() {
  Graph g;
  const auto gId0 = g.createSubGraphId("g0");
  const auto gId1 = g.createSubGraphId("g1");
  g.insert({}, 0, gId0, "");
  auto opId0 = g.insert({}, 0, gId1, "");
  g.insert({}, 0, gId0, "");
  auto opId1 = g.insert({}, 0, gId1, "");
  g.insert({}, 0, gId0, "");
  g.constraint(opId0, opId1);
  confirmSame(g, g.vanillaSchedule(gId1), {opId0, opId1});
  confirmSame(g, g.vanillaSchedules().at(gId1.get_u64()), {opId0, opId1});
}

void binConstraints0(uint64_t nBins, uint64_t nOps) {

  Graph g;

  std::vector<OpIds> bins(nBins);
  auto sgId = g.createSubGraphId("g0");

  std::vector<uint64_t> opToBin(nOps);

  for (uint64_t i = 0; i < nOps; ++i) {
    auto binId = (7 * i) % nBins;
    bins[binId].push_back(g.insert({}, 0, sgId, "op" + std::to_string(i)));
    opToBin[i] = binId;
  }
  g.binConstraint(bins);
  auto schedule = g.vanillaSchedule();

  std::vector<uint64_t> scheduleToBin;
  for (auto opId : schedule) {
    if (opId < nOps) {
      scheduleToBin.push_back(opToBin[opId.get()]);
    }
  }
  auto sorted = scheduleToBin;
  std::sort(sorted.begin(), sorted.end());
  if (sorted != scheduleToBin) {
    std::ostringstream oss;
    oss << "Failure to ensure that bin constraints are "
        << "satisfied while scheduling. "
        << "This with " << nBins << " bins and " << nOps << " Ops. ";
    throw poprithms::test::error(oss.str());
  }
}

void assertInOrder(uint64_t start, uint64_t end, const OpIds &schedule) {
  auto current = start;
  for (auto x : schedule) {
    if (x == current) {
      ++current;
    }
  }
  if (current < end) {
    std::ostringstream oss;
    oss << "the values in [start=" << start << ", end=" << end
        << ") do not appear in order in " << schedule;
    throw poprithms::test::error(oss.str());
  }
}

void assertNotInOrder(uint64_t start, uint64_t end, const OpIds &schedule) {
  auto current = start;
  for (auto x : schedule) {
    if (x == current) {
      ++current;
    }
  }
  if (current >= end) {
    std::ostringstream oss;
    oss << "the values in [start=" << start << ", end=" << end
        << ") do appear in order in " << schedule;
    throw poprithms::test::error(oss.str());
  }
}

void toggleEager0() {

  auto getGraph = [](uint64_t nOps,
                     const std::vector<uint64_t> &toggleTimes) {
    Graph g;
    auto gId = g.createSubGraphId("g0");

    std::vector<bool> tTimes(nOps, false);
    for (auto t : toggleTimes) {
      tTimes[t] = true;
    }

    for (uint64_t i = 0; i < nOps; ++i) {
      if (tTimes[i]) {
        g.toggleEager(gId, !g.eagerIsEnabled(gId));
      }
      g.insert({}, 0, gId, "");
    }
    return g;
  };

  {
    auto schedule = getGraph(30, {/* on */ 10, /* off */ 15, /* on */ 20})
                        .randomSchedule(1011);
    assertInOrder(10, 15, schedule);
    assertInOrder(20, 30, schedule);
  }
  {
    const auto schedule = getGraph(40,
                                   {
                                       0,
                                       3, // off
                                       6,
                                       11, // off
                                       15,
                                       19 // off
                                   })
                              .randomSchedule(1053);
    assertInOrder(0, 3, schedule);
    assertInOrder(6, 11, schedule);
    assertInOrder(15, 19, schedule);
    assertNotInOrder(20, 40, schedule); // exceedingly low probability here.
  }
}

void ensureLastOf0() {

  Graph g;
  auto gId = g.createSubGraphId("g0");
  for (uint64_t i = 0; i < 10; ++i) {
    g.insert({}, 0, gId, "");
  }
  g.ensureLastOfCurrentOps(OpId(5));
  if (g.randomSchedule(1011).back() != OpId(5)) {
    throw poprithms::test::error(
        "Op 5 should be at the back, failure of ensureLastOfCurrentOps");
  }

  bool caught{false};
  try {
    g.ensureLastOfCurrentOps(OpId(3));
    g.randomSchedule(1053);
  } catch (const poprithms::error::error &) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error("Should have detected a cylce");
  }
}

void mayBeFinals0() {

  Graph g;
  auto gId = g.createSubGraphId("g0");

  const auto a = g.insert({}, 1, gId, "");
  const auto b = g.insert({}, 1, gId, "");
  const auto c = g.insert({{a, 0}}, 0, gId, "");
  const auto d = g.insert({}, 1, gId, "");
  g.constraint(b, d);
  auto mays = g.mayBeFinals(gId);
  std::sort(mays.begin(), mays.end());
  if (mays != OpIds{c, d}) {
    throw poprithms::test::error("c and d are the 2 Ops which have to "
                                 "potentially to be scheduled last");
  }
}

void tensorIds0() {
  Graph g;
  auto gId0    = g.createSubGraphId("g0");
  auto gId1    = g.createSubGraphId("g1");
  const auto a = g.insert({}, 1, gId0, "");
  const auto b = g.insert({}, 1, gId1, "");
  const auto c = g.insert({{a, 0}}, 2, gId0, "");
  const auto d = g.insert({}, 1, gId1, "");
  auto in0     = g.tensorIds(gId0);
  std::sort(in0.begin(), in0.end());
  auto in1 = g.tensorIds(gId1);
  std::sort(in1.begin(), in1.end());

  if (in0 != std::vector<TensorId>{{a, 0}, {c, 0}, {c, 1}}) {
    throw poprithms::test::error("expected outs of a and c in sub-graph 0");
  }

  if (in1 != std::vector<TensorId>{{b, 0}, {d, 0}}) {
    throw poprithms::test::error("expected outs of b and d in sub-graph 1");
  }
}

} // namespace

int main() {

  basic0();

  basic1();

  binConstraints0(4, 16);
  // Sparse bins, definitely some empty ones:

  binConstraints0(30, 10);

  toggleEager0();

  ensureLastOf0();

  mayBeFinals0();

  tensorIds0();

  return 0;
}
