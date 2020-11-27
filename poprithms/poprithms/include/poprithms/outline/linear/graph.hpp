// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_OUTLINE_LINEAR_GRAPH
#define POPRITHMS_OUTLINE_LINEAR_GRAPH

#include <array>
#include <functional>
#include <string>
#include <vector>

#include <poprithms/outline/linear/linearusings.hpp>
#include <poprithms/outline/linear/op.hpp>
#include <poprithms/outline/linear/tensor.hpp>
#include <poprithms/schedule/supercon/graph.hpp>

namespace poprithms {
namespace outline {
namespace linear {

// Algo0 and Algo1 brought in from popart/subgraph TODO(T19425)
// Algo2 is a proof-of-concept algorithm to check basic functionality.
// As part of TODO(T19567) a new algorithm will be added.
enum class OutliningAlgorithm { Algo0, Algo1, Algo2, N };
enum class SchedulingAlgorithm { Filo, N };

std::ostream &operator<<(std::ostream &, OutliningAlgorithm);
std::ostream &operator<<(std::ostream &, SchedulingAlgorithm);

// An Outline object contains the nested sub-graphs found by an outlining
// algorithm
class Outline {
public:
  using Subgraph = std::vector<OpId>;
  using Match    = std::vector<Subgraph>;
  using Matches  = std::vector<Match>;

  Outline(const Matches &m, const uint64_t N) : allMatches(m), nOps_(N) {}

  uint64_t nMatches() const { return allMatches.size(); }
  const Match &match(uint64_t i) { return allMatches[i]; }
  const Matches &matches() const { return allMatches; }
  uint64_t nOps() const { return nOps_; }

private:
  // The outlined groups, which correspond to PopART's current Matches.
  // Example: If matches[9] = {{0,1}, {3,4}, {7,8}} then  {0,1}, {3,4} and
  // {7,8} are all equivalent subgraphs, and can have a single CallOp to
  // reduce code duplication
  const Matches allMatches;

  // The number of Ops in the Graph outlined
  const uint64_t nOps_;
};

// Note 1: If a specific schedule is desired, then constraints can be inserted
// to ensure that it is exactly reproduced.
// Note 2: There is no direct connection between constraints and input/output
// Tensors.
class Graph {
public:
  TensorId insertTensor(const Shape &, DType, const std::string &dbs);
  OpId insertOp(Color, Type, const std::string &debugString);

  // ensure "from" is scheduled before "to"
  void insertConstraint(OpId from, OpId to);
  bool containsConstraint(OpId from, OpId to) const {
    return get(from).hasOpOut(to);
  }

  // Input and output Tensors. Op constraints must be inserted separately
  //                  =======
  void insertIn(OpId, InIndex, TensorId);
  void insertOut(OpId, OutIndex, TensorId);

  // ensure that the Ops in "g" are contiguous, so that for any a \in g and b
  // \in g, there is no c not \in g s.t. a before c and c before b. Bins can
  // only be inserted after all Ops and constraints have been inserted.
  // TODO(T19634) implement this
  // void insertBin(const std::vector<OpId> &g);

  // ensure that "a" before "b" iff "c" before "d"
  void insertOrderCouple(OpId a, OpId b, OpId c, OpId d);

  // This function will attempt to minimize
  //  CostOfCalls + CostOfOps
  //
  // where
  //  CostOfCalls =
  //  sum_{all Matches m} [
  //    sum_{all Subgraphs s : m} [
  //       (sum_{external input Tensor t in s} (copyCost(t.size)) +
  //       (sum_{external output Tensor t in s} (copyCost(t.size))
  //     ]
  //  ]
  //
  // and
  //  CostOfOps =
  //  sum_{all "leaf Ops" o} [ opCost(o.type, o.ins) ]
  //
  // where "leaf Ops" are, roughly speaking, all Ops for which "code" is
  // generated.
  //
  // More specifically, the CostOfOps term is careful to not double count for
  // equivalent Ops in distinct Subgraphs of a Match. This is the
  // main difference between the CostOfOps term and the CostOfCalls term.
  //
  // To this end, suppose that the Subgraphs in a Match m are ordered with
  // indices 0 through m.size -1. An Op is defined to be a leaf Op if either
  //  1) it is not in any Match,
  //  2) the smallest Subgraph in which it appears has index 0.
  //
  //  Here is an example for a Graph with Ops
  //                  [a0, b0, c0, a1, b1, c1, a2, b2, c2, d, a3],
  // where the 3 types are a, b, c, and d, and there are 4, 3, 2 and 1
  // instances of each, respectively. Suppose that,
  //
  //  Matches = {{{a0, b0, c0}, {a1, b1, c1}},
  //             {{a1, b1}, {a0, b0}, {a2, b2}}}.
  //
  //  The leaf Ops are:
  //   d  - as it is not in any Match
  //   a3  - "  "  "  "   "  "   "
  //   a1 - the smallest Subgraph it appears in (a 2 Op Subgraph) has index 0
  //   b1 -  "      "        "     "     "   "   " " "      "      "    "   "
  //   a0 -  "      "        "     "     "   "   " 3 "      "      "    "   "
  //
  // The leaf Ops correspond to what we expect for "code" cost counting:
  // CostOpOps = 2*opCost(a, .) + opCost(b, .) + opCost(c, .) + opCost(d, .)
  //
  //
  // The functions copyCost and opCost are user defined.
  //
  // One shortcoming with this API is that copyCost and opCost cannot capture
  // any information about poplar Tensor layouts, which is very important in
  // determining memory requirement.
  //
  Outline getOutline(
      // Map an Op with known input and output shapes and types to a code size
      // cost. Example : If the cost of a matmul with float32 inputs of shapes
      // (8,6) and (6,4) is 1.0, this would correspond to opCodeSize(MatMul,
      // {{8,6}, {6,4}}) = 1.0.
      const std::function<
          double(Type, const std::vector<std::tuple<Shape, DType>> &inTens)>
          &opCost,

      // The cost of copying a Tensor of a certain size. Example : If the cost
      // of copying a Tensor with size 100 bytes is 1.0, then copyCost(100)
      // = 1.0
      const std::function<double(uint64_t)> &copyCost,

      // Are 2 subgraphs required to have the same external inputs and outputs
      // to be considered Subgraphs in a Match?
      //
      // The current PopART outliner has "true" for both inputs and outputs.
      // T17667 is to relax this constraint. Example:
      //
      // Subgraph 1:
      //  A-->B-->C-->E
      //
      // Subgraph 2:
      //  A-->B-->C-->F
      //       \.
      //        G
      //
      // If requireCommonExternalOutputs is true, (A,B,C) is not matched
      // between the 2 subgraphs, as subgraph 1 has external output E, while
      // subgraph 2 has extranl outputs F and G
      bool requireCommonExternalInputs,
      bool requireCommonExternalOutputs,

      OutliningAlgorithm outliningAlgorithm,
      SchedulingAlgorithm schedulingAlgorithm

      // more options here for time allowed, etc.
  );

  uint64_t nTensors() const { return allTensors.size(); }
  uint64_t nOps() const { return allOps.size(); }
  const Op &get(OpId id) const { return allOps[id.get()]; }
  const Tensor &get(TensorId id) const { return allTensors[id.get()]; }

  uint64_t nColors() const;
  uint64_t nTypes() const;

  void append(std::ostream &ost) const;

  // TODO(T19635) this function will set canonical color and type
  void finalize();

private:
  Op &get(OpId id) { return allOps[id.get()]; }
  Tensor &get(TensorId id) { return allTensors[id.get()]; }

  std::vector<Op> allOps;
  std::vector<Tensor> allTensors;

  void setSchedule(SchedulingAlgorithm);
  std::vector<OpId> schToOp;
  std::vector<ScheduleIndex> opToSch;

  std::vector<std::array<OpId, 4>> orderCouples;

  std::vector<std::vector<uint64_t>> getEdges_u64() const;
  schedule::supercon::Couples getOrderCouples_u64() const;

  // TODO(T19636) : canonical color and type
  bool isFinalized{false};
};

std::ostream &operator<<(std::ostream &, const Graph &);

} // namespace linear
} // namespace outline
} // namespace poprithms

#endif
