// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <sstream>

#include <poprithms/memory/unwind/error.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/solution.hpp>

using namespace poprithms::memory::unwind;

//  inner graph:
//
//       (10,4) -> dimShuffle -> flatten -> (40,)
//
//
// outer graph:
//
//     (10,8) -+--> slice[:,0:4] -> call(inner) - x0 -+
//             |                                      +-- cat -> (80,)
//             +--> slice[:,4:8] -> call(inner) - x1 -+
//
// x0 has a target layout, with "attractionValue" attraction.
//
//
// Starting at x0, the different sinks get their Paths set with Algo::Greedy0.
//

void callWithCopies0(double attractionValue) {

  Graph g;

  const auto innerInput = g.sink({10, 4});
  g.setName(innerInput.opId(), "inner input");

  const auto ds = g.dimShuffle(innerInput, {{1, 0}});
  g.setName(ds.opId(), "dimShuffle");

  const auto innerOutput = g.flatten(ds);
  g.setName(innerOutput.opId(), "inner output");

  const auto outerInput = g.sink({10, 8});
  g.setName(outerInput.opId(), "outer input");

  const auto s0 = g.slice(outerInput, {0, 0}, {10, 4});
  g.setName(s0.opId(), "slice0");

  const auto s1 = g.slice(outerInput, {0, 4}, {10, 8});
  g.setName(s1.opId(), "slice1");

  const auto x0 = g.call({s0}, {innerInput}, {innerOutput}, 1.0)[0];
  g.setName(x0.opId(), "x0 (call out)");

  const auto knownLayout = g.source(g.shape(x0));
  g.setName(knownLayout.opId(), "x0 target");

  g.insertValuedPair(x0, knownLayout, attractionValue);

  const auto x1 = g.call({s1}, {innerInput}, {innerOutput}, 1.0)[0];
  g.setName(x1.opId(), "x1 (call out)");

  const auto cat = g.concat({x0, x1}, 0);
  g.setName(cat.opId(), "concat");

  const Solution soln(g);

  std::ostringstream oss;
  oss << g << "\n\n\n";
  if (soln.inwardsPaths(x0) != Paths{Path(knownLayout, Chain({40}), x0)}) {
    oss << "\"x0\", should be exactly like \"knownLayout\" "
        << "due to inserted attractor pair. Therefore we expected an "
        << "Identity Chain. ";
    throw error(oss.str());
  }

  if (soln.inwardsPaths(innerOutput) !=
      Paths{Path(knownLayout, Chain({40}), innerOutput)}) {
    oss << "\"innerOutput\" should have the same layout as "
        << "\"x0\", to soln.t copy-out-of-call points. ";
    throw error(oss.str());
  }

  auto c0 = Chain({40});
  c0.reshape({4, 10});
  c0.dimShuffle({{1, 0}});
  if (soln.inwardsPaths(innerInput) !=
      Paths{Path(knownLayout, c0, innerInput)}) {
    oss << "\"innerOutput\" should have the layout of " << x0
        << ", unwound be reshaping and dimshuffling.";
    throw error(oss.str());
  }

  auto c01 = c0;
  c01.settFillInto({0, 0}, {0, 4});
  auto c02 = c0;
  c02.settFillInto({0, 4}, {0, 0});
  Path p0(knownLayout, c01, outerInput);
  Path p1(knownLayout, c02, outerInput);
  if (soln.inwardsPaths(outerInput) != Paths{p0, p1} &&
      soln.inwardsPaths(outerInput) != Paths{p1, p0}) {
    oss << "\"outerInput\" should have its layout determined by "
        << "the target to which its slices are copied, in the inner graph. ";
    throw error(oss.str());
  }

  Chain cat0({40});
  cat0.settFillInto(Lower{40}, Upper{0});
  Chain cat1({40});
  cat1.settFillInto(Lower{0}, Upper{40});
  p0 = Path(knownLayout, cat0, cat);
  p1 = Path(knownLayout, cat1, cat);
  if (soln.inwardsPaths(cat) != Paths{p0, p1} &&
      soln.inwardsPaths(cat) != Paths{p1, p0}) {
    oss << "\"outerOutput\" should have its layout determined by "
        << "by concatening x0, which has a known layout. ";
    throw error(oss.str());
  }

  const auto elmsPerTensor = 40;
  const auto expectedScore =
      elmsPerTensor * 4 // This for the 2 copies in, and the 2 copies out
      + attractionValue *
            elmsPerTensor; // This for getting a match to the known layout.

  const auto observedScore = soln.getScore();

  if (expectedScore != observedScore) {
    std::ostringstream oss2;
    oss2 << "score with attraction value of " << attractionValue << " = "
         << soln.getScore() << ". Expected " << expectedScore << ". ";
    throw error(oss2.str());
  }
}

int main() {
  callWithCopies0(0.1);
  callWithCopies0(10.0);
  return 0;
}
