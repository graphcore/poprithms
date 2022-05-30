// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <ostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>

#include <poprithms/common/multiout/consumptionid.hpp>
#include <poprithms/common/multiout/graph.hpp>
#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/op.hpp>
#include <poprithms/common/multiout/optionaltensorid.hpp>
#include <poprithms/common/multiout/optraversal.hpp>
#include <poprithms/common/multiout/skiptraversal.hpp>
#include <poprithms/common/multiout/traversal.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/stringutil.hpp>

namespace {
using namespace poprithms::common;
using Shape = poprithms::ndarray::Shape;
using poprithms::common::multiout::ContiguousInIndexSubset;
using poprithms::common::multiout::ContiguousOutIndexSubset;
using poprithms::common::multiout::InIndex;
using poprithms::common::multiout::InIndices;
using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OpIds;
using poprithms::common::multiout::OptionalTensorId;
using poprithms::common::multiout::OptionalTensorIds;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using Shapes = poprithms::ndarray::Shapes;

namespace test {

class Op : public multiout::Op {
public:
  Op(const multiout::Op::State &s) : multiout::Op(s) {}
  std::string typeString() const final { return "LazyMauveOp"; }
  std::unique_ptr<multiout::Op> cloneMultioutOp() const final {
    return std::make_unique<Op>(*this);
  }

private:
  bool multiOutTypeSpecificEqualTo(const multiout::Op &) const final {
    return true;
  };
};

class Graph : public multiout::Graph {

public:
  multiout::OpId insert(const TensorIds &inIds, uint64_t nOuts) {
    const Shapes outShapes(nOuts, Shape({1}));
    std::vector<multiout::ConsumptionIds> consOut(nOuts);
    const multiout::Op::State s(nOps(), inIds, consOut, outShapes, "", *this);
    return insertMultioutOp(std::make_unique<test::Op>(s));
  }

  using multiout::Graph::removeInputs;
  using multiout::Graph::removeOutputs;
  virtual ~Graph() override = default;
  void appendOpColumns(std::ostream &, const OpIds &) const final;

private:
  void verifyMultioutDerivedGraphValid() const final {}
  void verifyMultioutDerivedOpValid(OpId) const final {}

  bool multiOutTypeSpecificEqualTo(const multiout::Graph &) const final {
    return true;
  }

  void multiOutTypeSpecificRemoveOp(OpId, const OptionalTensorIds &) final {}

  void
  multiOutTypeSpecificVerifyValidSubstitute(const TensorId &,
                                            const TensorId &) const final {}

  void
  multiOutTypeSpecificRemoveInputs(OpId,
                                   const ContiguousInIndexSubset &) final {}

  void multiOutTypeSpecificRemoveOutputs(OpId,
                                         const ContiguousOutIndexSubset &,
                                         const OptionalTensorIds &) final {}
};

} // namespace test

void testOutConsumers0() {
  test::Graph g;
  auto a = g.insert({}, 5);
  auto b = g.insert({{a, 0}}, 2);
  auto c = g.insert({{a, 0}}, 2);
  auto d = g.insert({{a, 2}, {b, 0}, {c, 1}}, 1);
  (void)d;

  // a is consumed at indices 0 and 2.

  auto observed = g.outIndicesConsumed(a);
  std::sort(observed.begin(), observed.end());

  if (observed.size() != 2 || observed[0] != 0 || observed[1] != 2) {
    std::ostringstream oss;
    oss << "Expected {0,2} as the consumed output tensors";
    throw poprithms::test::error(oss.str());
  }
}

void test0() {
  test::Graph g;
  g.setName("my_test_graph");
  if (g.getName() != "my_test_graph") {
    throw poprithms::test::error(
        "Failed to correctly set and get name in test::Graph class. ");
  }

  multiout::OpIds collected;
  for (uint64_t i = 0; i < 50; ++i) {
    collected.push_back(g.insert({}, 0));
  }
  if (collected[34] != multiout::OpId(34)) {
    throw poprithms::test::error(
        "Expected OpIds to increment by 1, starting at 0");
  }
}

void test::Graph::appendOpColumns(std::ostream &ost, const OpIds &ids) const {
  const auto c = getMultioutColumns(ids, {});
  ost << alignedColumns(c);
}

std::ostream &operator<<(std::ostream &ost, const test::Graph &g) {
  g.append(ost);
  return ost;
}

void testLogging0() {

  //  OpId OpType      InTensors OutIndex Shape
  //  ---- ------      --------- -------- -----
  //  0    LazyMauveOp ()
  //  1    LazyMauveOp ()        0        (1)
  //                             1        (1)
  //                             2        (1)
  //  2    LazyMauveOp ()        0        (1)
  //  3    LazyMauveOp ()

  test::Graph g;
  g.insert({}, 0);
  g.insert({}, 3);
  g.insert({}, 1);
  g.insert({}, 0);
  std::cout << g << std::endl;

  const auto outCols = g.getMultioutColumns({});
  if (outCols.empty()) {
    throw poprithms::test::error("No multiout columns in test");
  }
  for (auto x : outCols) {
    if (x.nEntries() != outCols[0].nEntries()) {
      throw poprithms::test::error("size of each column should be the same");
    }
  }

  // expect 2 blanks in Shape column:
  for (auto col : outCols) {
    auto vs = col.entries();
    if (std::any_of(
            vs.cbegin(), vs.cend(), [](auto s) { return s == "Shape"; })) {

      auto isSpace = [](std::string s) {
        return std::all_of(
            s.cbegin(), s.cend(), [](auto x) { return std::isspace(x); });
      };

      int cnt = std::accumulate(
          vs.cbegin(),
          vs.cend(),
          int(0),
          [&isSpace](int a, const std::string &b) { return a + isSpace(b); });

      if (cnt != 2) {
        throw poprithms::test::error(
            "Expected 2 empty rows in the Shape column");
      }
    }
  }
}

void testInsAndOuts() {

  test::Graph g;
  auto a = g.insert({}, 2);
  auto b = g.insert({}, 3);
  auto c = g.insert({{a, 0}, {b, 1}, {b, 2}}, 4);

  auto insNouts = g.inAndOutTensorIds(c);

  std::sort(insNouts.begin(), insNouts.end());

  if (insNouts !=
      TensorIds({{a, 0}, {b, 1}, {b, 2}, {c, 0}, {c, 1}, {c, 2}, {c, 3}})) {
    throw poprithms::test::error("Incorrect input+output TensorIds");
  }
}

void testHashTensorId() {
  std::unordered_set<uint64_t> nDistintHashes;
  uint64_t nTensors{10000};

  std::mt19937_64 gen(1011);
  for (uint64_t i = 0; i < nTensors; ++i) {

    TensorId tId;
    auto k = gen() % 3;

    // random OpId and OutIndex
    if (k == 0) {
      tId = TensorId(static_cast<uint32_t>(gen()),
                     static_cast<uint32_t>(gen()));
    }

    // repeated OpIds, random OutIndex
    else if (k == 1) {
      tId = TensorId(gen() % 3, static_cast<uint32_t>(gen()));
    }

    // repeates OutIndex, random OpIds
    else {
      tId = TensorId(static_cast<uint32_t>(gen()), gen() % 3);
    }
    nDistintHashes.emplace(std::hash<TensorId>{}(tId));
  }

  std::cout << nDistintHashes.size() << std::endl;

  if (static_cast<double>(nDistintHashes.size()) /
          static_cast<double>(nTensors) <
      0.99) {
    throw poprithms::test::error("Failed hash test");
  }
}
void testTraversal0() {
  test::Graph g;
  auto a = g.insert({}, 2);
  auto b = g.insert({}, 2);

  // 4 ins, 2 outs: 8 paths through this op
  auto c = g.insert({{a, 0}, {a, 1}, {b, 0}, {b, 1}}, 2);

  // 4 ins, 3 outs: 12 paths through this op
  auto d = g.insert({{a, 0}, {a, 1}, {b, 0}, {b, 1}}, 3);

  // 4 ins, 5 outs: 20 paths through this op
  g.insert({{a, 0}, {b, 0}, {c, 0}, {d, 0}}, 5);

  // 40 paths in total:
  if (multiout::depthFirstForward(
          g, {{a, 0}, {a, 1}, {b, 0}, {b, 1}}, [](auto) { return true; })
          .size() != 40) {
    throw poprithms::test::error("Expected 40 OpTraversals");
  }

  if (multiout::depthFirstForward(
          g, {{a, 0}, {a, 1}, {b, 0}, {b, 1}}, [](auto) { return false; })
          .size() != 0) {
    throw poprithms::test::error("Expected 0 OpTraversals");
  }

  if (multiout::depthFirstForward(g,
                                  {{a, 0}, {a, 1}, {b, 0}, {b, 1}},
                                  [](auto opTraversal) {
                                    return opTraversal.outIndex().get() % 2 ==
                                           0;
                                  })
          .size() != 24) {
    // 4 through c (all to (c,0)).
    // 8 through d (all to (d,0) and (d,2)).
    // 12 through e.
    throw poprithms::test::error(
        "Expected 24 OpTraversals: 4 through c, 8 through "
        "d and 12 through e.");
  }
}

void testTraversal1() {
  test::Graph g;

  auto a = g.insert({}, 1);
  auto b = g.insert({}, 1);
  auto c = g.insert({{a, 0}, {b, 0}}, 2);
  auto out =
      multiout::depthFirstForward(g, {{a, 0}, {b, 0}}, [](auto opTraversal) {
        return opTraversal.inIndex() == 0 && opTraversal.outIndex() == 1;
      });

  multiout::OpTraversal expected{0, c, 1};
  if (out != std::vector{expected}) {
    throw poprithms::test::error("Failed in basic traversal test");
  }
}

void verifyEdges(const test::Graph &g, const std::map<OpId, TensorIds> &ins) {

  // verify that consumers and outputs and inputs all agree.
  g.verifyValid();

  for (const auto &[op, inTensors] : ins) {
    if (g.nInTensors(op) != inTensors.size()) {
      std::ostringstream oss;
      oss << "Error verify graph after input/output removals. "
          << "Expected the number of inputs of " << op << " to be "
          << inTensors.size() << " but it is " << g.nInTensors(op);
      throw poprithms::test::error(oss.str());
    }
    for (InIndex inIndex = 0; inIndex < inTensors.size(); ++inIndex) {
      if (g.inTensorId(op, inIndex) != inTensors[inIndex.get()]) {
        std::ostringstream oss;
        oss << "Error verify graph after input/output removals. "
            << "Expected the input #" << inIndex << " of op " << op
            << " to be " << inTensors[inIndex.get()] << " but it is "
            << g.inTensorId(op, inIndex);
        throw poprithms::test::error(oss.str());
      }
    }
  }
}

void testRemoveEdges0() {
  {
    test::Graph g;

    /**
     * a ---+
     *      + ---> c
     * b ---+
     * */
    auto a = g.insert({}, 1);
    auto b = g.insert({}, 1);
    auto c = g.insert({{a, 0}, {b, 0}}, 1);

    /**
     * a
     *      + ---> c
     * b ---+
     * */
    g.removeInputs(c, {0});
    verifyEdges(g, {{a, {}}, {b, {}}, {c, {{b, 0}}}});

    /**
     * a ---+
     *      + ---> c
     * b
     * */
    g.removeOutputs(b, {0}, {TensorId{a, 0}});
    verifyEdges(g, {{a, {}}, {b, {}}, {c, {{a, 0}}}});
  }

  {
    test::Graph g;
    auto a  = g.insert({}, /* n-outputs */ 4);
    auto b  = g.insert({{a, 0}, {a, 2}}, 1);
    auto c  = g.insert({{a, 0}, {a, 1}}, 1);
    auto ar = g.insert({}, 2);

    g.removeOutputs(a, {0, 2}, {TensorId{ar, 1}, TensorId{ar, 0}});
    /**
     * b got outputs 0 and 2 of a, but they're both removed so now must get
     * outputs of ar.
     *
     *  c got outputs 0 and 1 of a, but output 0 is removed, so now gets one
     * output of ar instead. The output index of a for its second input
     * changes from 1 to 0.
     * */
    verifyEdges(
        g,
        {{a, {}}, {ar, {}}, {b, {{ar, 1}, {ar, 0}}}, {c, {{ar, 1}, {a, 0}}}});
  }

  {

    test::Graph g;

    /**
     *
     *   0+--     +
     *    |       +--- c (gets 0 and 1)
     *    |       +
     *    |
     *   1+---    +
     * a -+       +--- b (gets 0 and 2)
     *   2+---    +
     *    |
     *   3+---
     *
     *
     * Then outputs 0 and 2 are removed, and must be replaced by 1 and 3.
     *
     * So c gets 1 and 1 and b gets 1 and 3.
     *
     * Which when shifted down to fill in the gaps means
     * c gets 0 and 0 and b gets 0 and 1.
     *
     *  */

    auto a = g.insert({}, 4);
    auto b = g.insert({{a, 0}, {a, 2}}, 1);
    auto c = g.insert({{a, 0}, {a, 1}}, 1);
    (void)b;
    (void)c;
    g.removeOutputs(a, {0, 2}, {TensorId{a, 1}, TensorId{a, 3}});
    verifyEdges(g, {});
  }

  {
    // Longer example.
    test::Graph g;
    auto a = g.insert({}, 1);
    auto b = g.insert({{a, 0}, {a, 0}, {a, 0}}, 2);
    auto c = g.insert({{b, 0}, {b, 1}, {b, 0}}, 3);
    auto d = g.insert({{b, 1}, {c, 1}}, 0);
    g.removeOutputs(b, {0, 1}, {TensorId(a, 0), TensorId(a, 0)});
    verifyEdges(g,
                {{a, {}},
                 {b, {{a, 0}, {a, 0}, {a, 0}}},
                 {c, {{a, 0}, {a, 0}, {a, 0}}},
                 {d, {{a, 0}, {c, 1}}}});

    g.removeInputs(c, {1});
    g.removeInputs(d, {0});

    verifyEdges(g,
                {{a, {}},
                 {b, {{a, 0}, {a, 0}, {a, 0}}},
                 {c, {{a, 0}, {a, 0}}},
                 {d, {{c, 1}}}});

    auto e = g.insert({}, 2);
    g.removeOutputs(a, {0}, {TensorId(e, 0)});
    verifyEdges(g,
                {{a, {}},
                 {b, {{e, 0}, {e, 0}, {e, 0}}},
                 {c, {{e, 0}, {e, 0}}},
                 {d, {{c, 1}}}});

    g.removeOutputs(e, {0}, {TensorId(e, 1)});
    verifyEdges(g,
                {{a, {}},
                 {b, {{e, 0}, {e, 0}, {e, 0}}},
                 {c, {{e, 0}, {e, 0}}},
                 {d, {{c, 1}}}});

    bool caught{false};
    try {
      g.removeOutputs(e, {0}, {TensorId(b, 0)});
    } catch (const poprithms::error::error &) {
      caught = true;
    }
    if (!caught) {
      throw poprithms::test::error(
          "failed to catch error where output is used to replace input");
    }
  }
}

void testRemoveEdges1() {

  auto verifyMessage = [](const poprithms::error::error &e,
                          const std::vector<std::string> &frags) {
    std::string w{e.what()};
    for (const auto &f : frags) {
      if (w.find(f) == std::string::npos) {
        // if (std::find(w.cbegin(), w.cend(), f) == w.cend()) {
        throw poprithms::test::error("Looking for sub-string " + f + " in " +
                                     w + " but failed. ");
      }
    }
  };

  {
    test::Graph g;
    auto a = g.insert({}, 1);
    auto b = g.insert({}, 1);
    g.insert({{a, 0}, {b, 0}}, 1);
    bool caught{false};
    try {
      g.removeOutputs(a, {0}, {TensorId{a, 0}});
    } catch (const poprithms::error::error &e) {
      verifyMessage(e, {"Cannot use an output which is about to be removed"});
      caught = true;
    }
    if (!caught) {
      throw poprithms::test::error(
          "Failed to catch error of using about-to-be-deleted tensor as "
          "replacement for deleted tensor");
    }
  }

  {
    test::Graph g;
    auto a = g.insert({}, 1);
    auto b = g.insert({}, 1);
    g.insert({{a, 0}, {b, 0}}, 1);
    bool caught{false};
    try {
      g.removeOutputs(a, {1}, {TensorId{a, 0}});
    } catch (const poprithms::error::error &e) {
      verifyMessage(e, {"Invalid OutIndex"});
      caught = true;
    }
    if (!caught) {
      throw poprithms::test::error(
          "Failed to catch non-existant replacement tensor");
    }
  }

  {
    test::Graph g;
    auto a = g.insert({}, 1);
    auto b = g.insert({}, 1);
    g.insert({{a, 0}, {b, 0}}, 1);
    bool caught{false};
    try {
      g.removeOutputs(a, {0}, {OptionalTensorId{}});
    } catch (const poprithms::error::error &) {
      caught = true;
    }
    if (!caught) {
      throw poprithms::test::error("Failed to catch error of not providing a "
                                   "real replacement (Optional not set)");
    }
  }
}

void testTraversal2() {

  //                       +-- x1 --------------------+
  //                       |                          |
  //        +---- (op0) ---+-- x2 -- (op1) -- x3      |
  //        |                                         |
  //   x0 --+-----(op2) ---+-- x4                     |
  //                       |                          v
  //                       +-- x5 -- (op3) -- x6 -- (op4) -- x7
  //
  test::Graph g;
  TensorId x0{g.insert({}, 1), 0};
  auto op0 = g.insert({x0}, 2);
  /* auto op1 = */ g.insert({x0}, 1);
  auto op2 = g.insert({x0}, 2);
  auto op3 = g.insert({{op2, 1}}, 1);
  auto op4 = g.insert({{op0, 0}, {op3, 0}}, 1);

  {
    auto out = multiout::depthFirstBackward(
        g, {{op4, 0}}, [](auto) { return true; });

    std::vector<multiout::OpTraversal> expected;
    expected.push_back({0, op4, 0});
    expected.push_back({1, op4, 0});
    expected.push_back({0, op3, 0});
    expected.push_back({0, op0, 0});
    expected.push_back({0, op2, 1});
    std::sort(expected.begin(), expected.end());
    if (out != expected) {
      throw poprithms::test::error("failure in test of backwards traversal");
    }

    for (TensorId start : TensorIds({x0, {op3, 0}, {op0, 0}, {op2, 1}})) {
      if (!multiout::isFwdReachable(
              g, {start}, {op4, 0}, [](auto) { return true; })) {
        throw poprithms::test::error("failure in test of isFwdReachable, "
                                     "testTraversal2. Is reachable from " +
                                     start.str());
      }
    }

    for (TensorId start : TensorIds({{op0, 1}, {op2, 0}})) {
      if (multiout::isFwdReachable(
              g, {start}, {op4, 0}, [](auto) { return true; })) {
        throw poprithms::test::error(
            "failure in test of isFwdReachable, "
            "testTraversal2. Is NOT reachable from " +
            start.str());
      }
    }
  }
}

void testMovesAndCopies() {
  {
    // copy constructor
    std::unique_ptr<test::Graph> g = std::make_unique<test::Graph>();
    auto b                         = g->insert({}, 1);
    auto c                         = g->insert({{b, 0}}, 1);
    auto g1                        = *g;
    g1.verifyOpsConnectedToThisGraph();
    g.reset(nullptr);
    auto ins1 = g1.inTensorIds(c);
  }
  {
    // move constructor
    test::Graph g;
    g.verifyOpsConnectedToThisGraph();
    for (uint64_t i = 0; i < 5; ++i) {
      g.insert({}, 1);
    }
    auto g1 = std::move(g);
    g1.verifyOpsConnectedToThisGraph();
  }

  // copy assignment operator
  {
    test::Graph g;
    for (uint64_t i = 0; i < 7; ++i) {
      g.insert({}, 1);
    }
    test::Graph g2;
    g2 = g;
    g2.verifyOpsConnectedToThisGraph();
  }

  // move assignment operator
  {
    test::Graph g;
    for (uint64_t i = 0; i < 11; ++i) {
      g.insert({}, 1);
    }
    test::Graph g2;
    g2 = std::move(g);
    g2.verifyOpsConnectedToThisGraph();
  }
}

void testOptionalTensorIds0() {

  auto a = OptionalTensorId(TensorId(0, 0));
  auto b = OptionalTensorId(TensorId(0, 1));
  auto c = OptionalTensorId();
  auto d = c;
  auto e = OptionalTensorId(TensorId(0, 0));

  if (a == b || a == c || a != e) {
    throw poprithms::test::error("Failure comparing optional tensor a");
  }

  if (c == a || c != d) {
    throw poprithms::test::error("Failure comparing optional tensor c");
  }
}

class SkipEdges {
public:
  using Skips = std::vector<std::pair<TensorId, TensorId>>;
  SkipEdges(const Skips &skips_) : skips(skips_) {}

  bool isCarriedTo(const TensorId &tId) const {
    for (const auto &p : skips) {
      if (p.second == tId) {
        return true;
      }
    }
    return false;
  }

  bool isCarriedFrom(const TensorId &tId) const {
    for (const auto &p : skips) {
      if (p.first == tId) {
        return true;
      }
    }
    return false;
  }

  TensorId carriedTo(const TensorId &tId) const {
    for (const auto &p : skips) {
      if (p.first == tId) {
        return p.second;
      }
    }
    throw poprithms::test::error("not carried from");
  }

  TensorId carriedFrom(const TensorId &tId) const {
    for (const auto &p : skips) {
      if (p.second == tId) {
        return p.first;
      }
    }
    throw poprithms::test::error("not carried to");
  }

private:
  Skips skips;
};

void testSkipTraverse0() {

  //
  //       start
  //         |
  //  x0 --> x1 --> x2
  //
  //     <---------
  //     carry back
  //
  // so for
  // 0 iterations : nothing visited
  // 1 iteration  : x1 and x2 visited
  // >1 iterations :  x1 and x2 and x0 visited.
  //
  test::Graph g;
  auto x0 = g.insert({}, 1);
  auto x1 = g.insert({{x0, 0}}, 1);
  auto x2 = g.insert({{x1, 0}}, 1);
  SkipEdges se({{{x2, 0}, {x0, 0}}});
  TensorIds starts{{x1, 0}};
  const auto accept = [](const auto &) { return true; };
  {
    auto outs0 = depthFirstFwdWithSkips(se, g, starts, accept, 2);
    if (outs0.size() != 3) {
      throw poprithms::test::error(
          "loop back should go back to x0: x1 -> x2 -> (skip) -> x3");
    }
  }

  {

    auto outs0 = depthFirstFwdWithSkips(se, g, starts, accept, 0);
    if (outs0.size() != 0) {
      throw poprithms::test::error(
          "rpt is 0, so shouldn't visit any tensors");
    }
  }

  {
    auto outs0 = depthFirstFwdWithSkips(se, g, starts, accept, 1);
    if (outs0.size() != 2) {
      throw poprithms::test::error(
          "rpt is 1, so shouldn't visit the loop back tensor");
    }
  }
}

void testSkipTraverse1() {

  // graph with no forward edges, only back carries.
  // x0  x1  x2
  //  <---
  //     <----

  test::Graph g;
  auto x0 = g.insert({}, 1);
  auto x1 = g.insert({}, 1);
  auto x2 = g.insert({}, 1);
  SkipEdges se({{{x2, 0}, {x1, 0}}, {{x1, 0}, {x0, 0}}});
  TensorIds starts{{x2, 0}};
  const auto accept = [](const auto &) { return true; };

  for (uint64_t i = 0; i < 3; ++i) {
    auto outs0 = depthFirstFwdWithSkips(se, g, starts, accept, i);
    if (outs0.size() != i) {
      std::ostringstream oss;
      oss << "At rptCount=" << i << ", expected " << i
          << " tensors to be visited, not " << outs0.size();
      throw poprithms::test::error(oss.str());
    }
  }
}

void testSkipTraverse2() {

  // Test of (1) backwards traversal with skips, and (2) a really large repeat
  // count (does the search terminate once all tensors visited?)

  test::Graph g;

  // lhs[i] --+
  //          +--=======-- outs[i]
  // rhs[i] --+               |
  //                          |
  //                          |
  // carries to rhs[i+1] <----+
  //
  OpIds lhs;
  OpIds rhs;
  OpIds adds;
  OpIds deadends;
  for (uint64_t i = 0; i < 10; ++i) {
    lhs.push_back(g.insert({}, 1));
    rhs.push_back(g.insert({}, 1));
    deadends.push_back(g.insert({{lhs.back(), 0}}, 1));
    deadends.push_back(g.insert({{rhs.back(), 0}}, 1));
    adds.push_back(g.insert({{lhs.back(), 0}, {rhs.back(), 0}}, 1));
  }

  std::vector<std::pair<TensorId, TensorId>> ses;
  for (uint64_t i = 0; i < 9; ++i) {
    ses.push_back({{adds[i], 0}, {rhs[i + 1], 0}});
  }

  TensorIds starts{{adds.back(), 0}};
  const auto accept = [](const auto &) { return true; };
  uint32_t n        = -1;

  auto outs = depthFirstBwdWithSkips(SkipEdges(ses), g, starts, accept, n);
  for (auto s : {lhs, rhs, adds}) {
    for (auto x : s) {
      if (outs.count({x, 0}) != 1) {
        std::ostringstream oss;
        oss << "Expected all outputs of lhs, rhs, and adds to be visited";
        throw poprithms::test::error(oss.str());
      }
    }
  }
  for (auto x : deadends) {
    if (outs.count({x, 0}) != 0) {
      throw poprithms::test::error(
          "Expected no deadend tensors to be visited");
    }
  }
}

void testForwardEdgeMap0() {

  test::Graph g;

  // x0 --+
  //      +-- x5
  // x1 --+
  //      +-- x6
  // x2 --+
  //      +-- x7
  // x3 --+
  //      +-- x8
  // x4 --+
  //

  OpIds allIds;
  for (uint64_t i = 0; i < 5; ++i) {
    allIds.push_back(g.insert({}, 1));
  }
  for (uint64_t i = 0; i < 4; ++i) {
    allIds.push_back(g.insert({{allIds[i], 0}, {allIds[i + 1], 0}}, 2));
  }

  // forms a single connected component:
  {
    auto fm = g.getMultioutForwardEdgeMap_u64({4});
    if (fm.nOps() != 9) {
      throw poprithms::test::error(
          "Expected all 9 ops to in edge map (connected)");
    }
  }
  {
    {
      auto fm = g.getMultioutForwardEdgeMap_u64({4, 5});
      if (fm.nOps() != 9) {
        throw poprithms::test::error(
            "Expected all 9 ops to in edge map (connected)");
      }
    }
  }
}

void testForwardEdgeMap1() {

  test::Graph g;

  // component 0
  auto x0 = g.insert({}, 2);
  auto x1 = g.insert({{x0, 1}}, 3);

  // component 1
  auto x2 = g.insert({}, 1);
  auto x3 = g.insert({{x2, 0}}, 1);

  if (g.getMultioutForwardEdgeMap_u64({0}).nOps() != 2) {
    throw poprithms::test::error("Expected only 2 ops (component 0)");
  }
  if (g.getMultioutForwardEdgeMap_u64({0, 3}).nOps() != 4) {
    throw poprithms::test::error("Expected 5 ops (components 0 and 1)");
  }

  // bridge components 0 and 1
  g.insert({{x1, 0}, {x3, 0}}, 1);
  if (g.getMultioutForwardEdgeMap_u64({0}).nOps() != 5) {
    throw poprithms::test::error(
        "Components 0 and 1 are connected now, expected 5 ops here");
  }
}

void testOnPathTo0() {

  test::Graph g;

  auto x = g.insert({}, 1);
  auto y = g.insert({}, 1);
  (void)y;

  auto z0 = g.insert({{x, 0}}, 1);
  auto z1 = g.insert({{x, 0}}, 10);
  auto z2 = g.insert({{z0, 0}, {z1, 0}}, 1);

  auto onPath_ = g.onPathTo({{z2, 0}});
  std::set<TensorId> onPathSet{onPath_.cbegin(), onPath_.cend()};

  std::set<TensorId> expected{{z2, 0}, {z1, 0}, {z0, 0}, {x, 0}};

  if (onPathSet != expected) {
    std::ostringstream oss;
    oss << "Expected the tensors "
        << TensorIds(expected.begin(), expected.end())
        << " to be on the path, not " << onPath_;
    throw poprithms::test::error(oss.str());
  }
}

} // namespace

int main() {
  test0();
  testLogging0();
  testInsAndOuts();
  testHashTensorId();
  testTraversal0();
  testTraversal1();
  testTraversal2();
  testMovesAndCopies();
  testOutConsumers0();
  testOptionalTensorIds0();
  testRemoveEdges0();
  testRemoveEdges1();
  testSkipTraverse0();
  testSkipTraverse1();
  testSkipTraverse2();
  testForwardEdgeMap0();
  testForwardEdgeMap1();
  testOnPathTo0();
  return 0;
}
