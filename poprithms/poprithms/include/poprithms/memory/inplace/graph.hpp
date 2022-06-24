// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_GRAPH_HPP
#define POPRITHMS_MEMORY_INPLACE_GRAPH_HPP

#include <array>
#include <memory>
#include <sstream>
#include <vector>

#include <poprithms/common/multiout/consumptionid.hpp>
#include <poprithms/common/multiout/graph.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/memory/alias/graph.hpp>
#include <poprithms/memory/inplace/allowmultigatealias.hpp>
#include <poprithms/memory/inplace/checkparallelwriteable.hpp>
#include <poprithms/memory/inplace/constantpadding.hpp>
#include <poprithms/memory/inplace/crosslink.hpp>
#include <poprithms/memory/inplace/padding.hpp>
#include <poprithms/memory/inplace/proposal.hpp>
#include <poprithms/memory/inplace/result.hpp>
#include <poprithms/memory/inplace/tensormap.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/util/copybyclone.hpp>
#include <poprithms/util/permutation.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

using common::multiout::ConsumptionId;
using common::multiout::ConsumptionIds;
using common::multiout::InIndex;
using common::multiout::OpId;
using common::multiout::OpIds;
using common::multiout::OptionalTensorIds;
using common::multiout::OutIndex;
using common::multiout::TensorId;
using common::multiout::TensorIds;
using memory::nest::Region;
using ndarray::Dimension;
using ndarray::Dimensions;
using ndarray::Shape;
using poprithms::common::multiout::ContiguousInIndexSubset;
using poprithms::common::multiout::ContiguousOutIndexSubset;
using Lower = ndarray::Shape::Lower;
using Upper = ndarray::Shape::Upper;
using common::multiout::ConsumptionId;
using common::multiout::ConsumptionIds;
using common::multiout::InIndex;
using common::multiout::InIndices;
using common::multiout::OpId;
using common::multiout::OpIds;
using common::multiout::OutIndex;
using common::multiout::TensorId;
using common::multiout::TensorIds;
using memory::alias::BroadcastPadding;
using memory::nest::DisjointRegions;
using memory::nest::Region;
using ndarray::Dimension;
using ndarray::Dimensions;
using ndarray::Shape;
using ndarray::Shapes;
using ndarray::Stride;
using ndarray::Strides;
using util::Permutation;

class Op;
class AliasGate;

/**
 * This graph class extends the functionality of the alias::Graph class, by
 * adding concepts and algorithms related to computation.
 *
 * The extension uses the HAS-A design, as the IS-A approach does not work due
 * to limitations on the alias::Graph class. In particular, the alias::Graph
 * has a 1:1 correspondence between Nodes and Tensors, which means
 * multi-output Nodes are not possible with it.
 *
 * Almost all methods in this class which insert Tensors do not perform
 * allocations. That is, almost all outputs of an Op are aliases of the Op's
 * inputs. The 2 exceptions are,
 *
 * 1) aliasGate. This method takes N inputs, and creates one output whose
 *    Shape is inferred by numpy-broadcasting the N inputs. The output is
 *    optionally aliased to one of the inputs with the same Shape. So with N
 *    inputs of the same Shape, there are N + 1 AliasGate possibilities: one
 *    "closed" variant, and N "open" variants, which respectively alias one of
 *    the N inputs.
 *
 * 2) multi. This Op has N inputs and creates M outputs, of user specified
 *           Shapes. How inputs and outpus are aliased, if at all, is also
 *           user specified.
 * */
class Graph : public common::multiout::Graph {

public:
  virtual ~Graph() override;
  Graph()                         = default;
  Graph(Graph &&)                 = default;
  Graph(const Graph &)            = default;
  Graph &operator=(Graph &&)      = default;
  Graph &operator=(const Graph &) = default;

  void verifyMultioutDerivedGraphValid() const final {}
  void verifyMultioutDerivedOpValid(OpId) const final {}

  /** Subsample a Tensor in a specified Region. \sa Region */
  TensorId settSample(const TensorId &, const Region &);

  /** Reverse a Tensor along certain dimensions. */
  TensorId reverse(const TensorId &, const Dimensions &);

  /** Reshape a Tensor, keeping the number of elements unchanged. */
  TensorId reshape(const TensorId &, const Shape &);

  /** Expand a Tensor, broadcasting in singleton dimensions. */
  TensorId expand(const TensorId &, const Shape &);

  /** Permute the dimensions a Tensor. */
  TensorId dimShuffle(const TensorId &, const Permutation &);

  /** Concatentate Tensors along a certain dimension */
  TensorId concat(const TensorIds &, uint64_t);

  /** Slice a Tensor in a region defined by lower and upper bounds. */
  TensorId slice(const TensorId &, const Lower &, const Upper &);

  /** Subsample a Tensor along a single dimension. */
  TensorId subSample(const TensorId &, Stride, Dimension);

  /** Reshape a Tensor to be of rank 1. */
  TensorId flatten(const TensorId &);

  /** Subsample a Tensor with different strides along all dimensions. */
  TensorId subSample(const TensorId &, const Strides &);

  /** Allocate a constant Tensor in this Graph */
  TensorId constant(const Shape &);

  /** Allocate a variable Tensor in this Graph */
  TensorId variable(const Shape &);

  /** Modify the elements of a Tensor, return alias. */
  TensorId modify(const TensorId &);

  /** Pad a Tensor, inserting constant/variable Tensor(s) below and above. */
  TensorId pad(const TensorId &,
               const LowerPadding &,
               const UpperPadding &,
               ConstantPadding,
               BroadcastPadding);

  /** Pad a Tensor below and above, with padding which is either parallel
   * writeable or is not. */
  TensorId pad(const TensorId &,
               const std::array<std::vector<int64_t>, 2> &lowerAndUpper,
               bool paddingIsParallelWriteable);

  /** A multi-purpose operation. The definition of which outputs modify,
   * alias, and use which inputs is defined by #mapping. */
  OpId multi(const TensorIds &inputs,
             const Shapes &outputShapes,
             const CrossLinks &mapping);

  /**
   * An aliasGate represents a variadic elementwise numpy-broadcast operation,
   * where the output may optionally alias, but not modify, one of the inputs.
   * In other words:
   * 1) it has N inputs and 1 output whose Shape is inferred  by
   *    numpy-broadcasting the N inputs.
   * 2) The output is optionally aliased to one of the inputs with the same
   *    Shape. If the N inputs all have the same Shape, there are N + 1
   *    AliasGate variants: one "closed" variant, and N "open" variants, which
   *    each respectively alias one of the N inputs.
   *
   * This method creates a closed (non-aliasing) aliasGate operation in this
   * Graph. In terms of aliasing, it is equivalent to allocating a new
   * variable Tensor.
   * */
  TensorId aliasGate(const TensorIds &);

  /** An open AliasGate, where the output is aliased to the \a i'th input. The
   * Shape of the i'th input must be same as the output Shape.
   * https://numpy.org/doc/stable/user/basics.broadcasting.html */
  TensorId aliasGate(const TensorIds &input, InIndex i);

  /**
   * \return true if the Op #id is an AliasGate, false otherwise.
   * */
  bool isAliasGate(OpId id) const;

  /**
   * AliasGate state queries. These methods throw errors if the OpId is not
   * an AliasGate.
   */
  bool aliasGateIsClosed(OpId) const;
  bool aliasGateIsOpen(OpId id) const { return !aliasGateIsClosed(id); }
  InIndex aliasGateInIndex(OpId) const;

  /** The ConsumptionIds of a Tensor which modify it. */
  ConsumptionIds modifiers(const TensorId &) const;

  /**
   * The consumers of #tId which read its value.
   * */
  ConsumptionIds readingConsumers(const TensorId &) const;

  /** All Tensors which are aliased to \a t */
  TensorIds allAliases(const TensorId &t) const;

  /**
   * \return true if the two Tensors \a a and \a b are aliased.
   * */
  bool areAliased(const TensorId &a, const TensorId &b) const;

  /**
   * \return true if all of the 'allocation' elements of \a sub and are also
   *         in \a super.
   * */
  bool contains(const TensorId &super, const TensorId &sub) const;

  /** Insert a topological constraint, ensuring that \a before appears before
   * \a after in all schedules. */
  void constraint(OpId before, OpId after);

  /** Insert a topological constraint between Tensor creators;
   * specifically, the Op which creates \a before must appear before the
   * creator of \a after in a schedule of this Graph.
   * */
  void constraint(const TensorId &before, const TensorId &after) {
    constraint(before.opId(), after.opId());
  }

  /** Insert a chain of topological constraints.
   * \a a -> \a b -> \a c -> ... */
  template <class Arg0, class... Args>
  void constraint(Arg0 a, Arg0 b, Args... c) {
    constraint(a, b);
    constraint(b, c...);
  }

  /** Insert topological constraints which ensure \a before appears before all
   * Ops in \a afters. */
  void constraints(OpId before, const OpIds &afters);

  /** Constraint that all Ops in \a befores appear before \a after. */
  void constraints(const OpIds &befores, OpId after);

  /** Insert multiple constraints */
  void constraints(const Constraints &);

  /**
   * Attempt to open an AliasGate at a specific InIndex.
   *
   * If the proposed change is accepted, the AliasGate is opened at the
   * proposed InIndex, and new constraints are inserted between Ops if
   * necessary. If the proposed opening is rejected, the proposed Op is
   * unchanged, and no constraints are inserted.
   *
   * \param proposal The proposed AliasGate to open, and the InIndex to open
   *                 at.
   *
   * \param checkParWrite Whether to disallow the opening if it results in
   *                      non-parallel writes.
   *
   * \param allowMultiGateAlias Whether to allow the input at the open index
   *                            of an alias gate to alias other inputs.
   *
   * \return The status of the attempt, describing whether or not the change
   *         took place. Possible failure Statuses:
   *
   * Cycle: Sometimes, opening an AliasGate results in new constraints
   *        between Ops, to ensure that Tensors are not modified too
   *        early, trashing memory which is used later. Sometimes, these
   *        constraints result in cycles, in which case the inplacing is
   *        rejected.
   *
   * NotParallelWriteable: Sometimes, opening an AliasGate results in a
   *                       Tensor which is not parallel writeable being
   *                       modified. If this happens, and \a checkParWri is
   *                       Yes, the proposal is rejected.
   *
   * AlreadyOpen: If the AliasGate is already open, the proposal is rejected.
   *
   * \see OpeningResult
   * \see Proposal
   * */
  OpeningStatus tryOpening(
      const Proposal &proposal,
      CheckParallelWriteable checkParWrite,
      AllowMultiGateAlias allowMultiGateAlias = AllowMultiGateAlias::Yes);

  /**
   * Attempt to open an AliasGate, without inserting final constraints and
   * without changing this Graph's representation. Only limited changes are
   * made. In pseudocode, the code flow might be:
   *
   * def tryOpening(.)
   *  result = tryOpeningPartial(.)
   *  if (result.valid):
   *    completeOpening(.);
   *
   * There are certain use cases (PopART) where one wants to leave an
   * AliasGate closed even after tryOpening has confirmed that it is valid.
   * This method makes such use cases possible.
   * */
  OpeningResult
  tryOpeningPartial(const Proposal &,
                    CheckParallelWriteable,
                    AllowMultiGateAlias = AllowMultiGateAlias::Yes);

  /** Perform final Graph modifications */
  void completeOpening(const OpeningResult &);

  /** Revert the changes made in tryOpeningPartial, if there were any */
  void backoutOpening(const Proposal &);

  /** Attempt Proposals in order, returning OpeningResults for each */
  OpeningStatuses tryOpenings(const Proposals &,
                              CheckParallelWriteable,
                              AllowMultiGateAlias = AllowMultiGateAlias::Yes);

  OpeningStatuses
  tryOpenings0(const TensorIds &,
               CheckParallelWriteable,
               AllowMultiGateAlias = AllowMultiGateAlias::Yes);

  OpeningStatuses
  tryOpenings0(const OpIds &,
               CheckParallelWriteable,
               AllowMultiGateAlias = AllowMultiGateAlias::Yes);

  /** Append a string describing this Graph to \a ost */
  void appendOpColumns(std::ostream &, const OpIds &) const final;

  /**
   * Consider a graph:
   *
   * a <- variable({})
   * b <- modify(a)
   * c <- modify(a),
   *
   *    variable
   *       |
   *    +--+-----+
   *    |        |
   * modify    modify.
   *
   * without any topological constraints between b and c, the final value of
   * 'a' is potentially ambiguous. Suppose that b is 'sqrt' and c is 'sin'.
   * The final value of 'a' depends on the schedule:
   *
   * (a,b,c) : a <- sin(sqrt(a))
   * (a,c,b) : a <- sqrt(sin(a)).
   *
   * As a second example, consider the 'compute' graph:
   *      --------------
   *
   * a <- variable({})
   * b <- variable({})
   * m <- sqrt_(a)
   * c <- a + x.
   *
   * Without any topological constraints between c and m, the final value of
   * 'c' is ambiguous.
   *
   * The general rule is as follows: If a tensor 'a' is consumed by an op 'm'
   * which modifies it, and 'a' is aliased to another tensor 'b' which is
   * consumed by an op 'c' which reads the value of 'b', then unless there
   * is a constraint between 'm' and 'c', the value of 'b' is ambiguous. By
   * 'reads' we include all ops which are not simply view-changers, or ops
   * like 'shape' which don't use the numerical values of the input. The case
   * of an op which resets the value of an input is not currently supported
   * TODO(T32975).
   *
   * An object of class 'AmbiguityStatus' is returned when ambiguity detection
   * is run on a Graph (see 'containsAmbiguity'). It,
   *
   * (1) records whether an ambiguity was detected, and if so
   * (2) records the first ambiguity detected.
   * */
  class AmbiguityStatus {
  public:
    /** Was an ambiguity detected? */
    bool detected() const { return detected_; }

    /**
     * If there was an ambiguity, what was the modifying Op involved?
     * */
    OpId modifier() const { return modifier_; }

    /**
     * If there was an ambiguity, which tensor was directy modified? (This
     * corresponds to 'a' in the second example above).
     * */
    TensorId modified() const { return modified_; }

    /**
     * If there was an ambiguity, which Op was the one which read the value of
     * the alias of the modified tensor?
     * */
    OpId reader() const { return reader_; }

    /**
     * If there was an ambiguity, which tensor was ambiguously read? (This
     * also corresponds to 'a' in the second example above).
     * */
    TensorId readIn() const { return readIn_; }

    /**
     * In the case where an ambiguity was detected, this string summarizes the
     * ambiguity.
     * */
    std::string summary() const { return summary_; }

    /**
     * Constructor for the case where there was an ambiguity.
     * */
    AmbiguityStatus(const Graph &g,
                    OpId modifier,
                    TensorId modified,
                    OpId reader,
                    TensorId readIn);

    static AmbiguityStatus None() { return {false}; }

  private:
    AmbiguityStatus(bool d) : detected_(d) {}

    std::string summary_;
    bool detected_;
    OpId modifier_;
    TensorId modified_;
    OpId reader_;
    TensorId readIn_;
  };

  /**
   * Perform ambiguity detection on this graph.
   * */
  AmbiguityStatus containsAmbiguity() const;

private:
  // Decoupling the true edges (constraints) from ambiguity detection allows
  // us to quickly determine if a graph is ambiguity free using a subset of
  // the edges in the graph.
  AmbiguityStatus
  containsAmbiguity(const std::vector<std::vector<uint64_t>> &edges) const;

  template <class T, class... Args>
  OpId
  createOp(const TensorIds &inIds, const Shapes &outShapes, Args... args);
  OpId insertOp(std::unique_ptr<Op>);

  bool
  multiOutTypeSpecificEqualTo(const common::multiout::Graph &) const final;

  void multiOutTypeSpecificRemoveOp(OpId, const OptionalTensorIds &) final {
    // TODO(T49501)
    unimplemented();
  }

  void
  multiOutTypeSpecificVerifyValidSubstitute(const TensorId &,
                                            const TensorId &) const final {
    // TODO(T49501)
    unimplemented();
  }

  Op &op(OpId);
  const Op &op(OpId) const;

  template <typename T> using Edges = std::vector<std::vector<T>>;

  // Get a simple edge map, which can be passed into an external scheduling
  // algorithm. Additional constraints can be included via 'additional'
  template <typename T>
  Edges<T> getFwdEdges(const Constraints &additional) const;

  // Get a simple edge map, which can be passed into an external scheduling
  // algorithm. Constraints can be removed by using the Condition filter.
  // The edge map includes all edges a->b for which cond(a, b) = true.
  template <typename T, typename Condition>
  Edges<T> getConditionalFwdEdges(Condition &&condition) const;

  std::vector<std::array<TensorId, 2>>
  createBroadcastPadElements(const Shape &,
                             const LowerPadding &,
                             const UpperPadding &,
                             ConstantPadding);

  std::vector<std::array<TensorId, 2>>
  createNonAliasedPadElements(const Shape &,
                              const LowerPadding &,
                              const UpperPadding &,
                              ConstantPadding);

  // The Tensor class is a thin API on top of the Graph class, to make user
  // code more succinct. The Graph and Tensor classes can be thought of as
  // interfaces to the same underlying graph structure.
  friend class Tensor;

  // Return true if the Constraints constraints are all already satisfied by
  // the current schedule.
  bool satisifedWithoutAnyChange(const Constraints &constraints) const;

  // This Graph class represents a DAG. The algorithm which opens AliasGates
  // uses a schedule, which is a linearization of the DAG, to ensure that no
  // cycles are created. This should remain private to this class.
  void setSchedule(OpIds &&);
  uint64_t scheduleIndex(OpId id) const;

  // Schedule and mapping from OpIds to schedule indices.
  std::vector<OpId> sched;
  std::vector<uint64_t> invSched;
  bool scheduleIsValid{false};

  // An Op in this Graph maps to 1 or more Nodes in an alias::Graph.
  alias::Graph aGraph_;
  alias::Graph &aGraph() { return aGraph_; }
  const alias::Graph &aGraph() const { return aGraph_; }
  TensorMap tensorMap;

  // Dynamically cast to an AliasGate.
  AliasGate &asAliasGate(OpId);
  const AliasGate &asAliasGate(OpId) const;

  void
  multiOutTypeSpecificRemoveInputs(OpId,
                                   const ContiguousInIndexSubset &) final {

    // TODO(T49501)
    unimplemented();
  }

  void multiOutTypeSpecificRemoveOutputs(OpId,
                                         const ContiguousOutIndexSubset &,
                                         const OptionalTensorIds &) final {

    // TODO(T49501)
    unimplemented();
  }
};

std::ostream &operator<<(std::ostream &, const Graph &);

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
