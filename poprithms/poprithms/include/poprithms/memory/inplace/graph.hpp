// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_GRAPH_HPP
#define POPRITHMS_MEMORY_INPLACE_GRAPH_HPP
#include <array>
#include <memory>
#include <vector>

#include <poprithms/memory/alias/graph.hpp>
#include <poprithms/memory/inplace/aliastype.hpp>
#include <poprithms/memory/inplace/consumer.hpp>
#include <poprithms/memory/inplace/crossalias.hpp>
#include <poprithms/memory/inplace/proposal.hpp>
#include <poprithms/memory/inplace/result.hpp>
#include <poprithms/memory/inplace/tensorid.hpp>
#include <poprithms/memory/inplace/tensormap.hpp>
#include <poprithms/memory/inplace/usings.hpp>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/permutation.hpp>
#include <poprithms/util/typedvector.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

/// Whether to check that all Tensors which are modified are parallel
/// writeable. This means that they contain no Constants, and contain no
/// self-aliases.
/// \see poplar::Tensor
enum class CheckParallelWriteable {
  No =
      0, ///< Allow Tensors which are not parallel writeable to be written to.
  Yes    ///< Only Tensors which are parallel writeable may be be modified,
         ///< ensure that no inplace transformations are applied which do not
         ///< satisfy this.
};
std::ostream &operator<<(std::ostream &, CheckParallelWriteable);

/// Whether to pad with a Variable or Constant.
/// see Graph::pad
enum class ConstantPadding {
  No = 0, ///< Pad with Variable Tensor(s)
  Yes     ///< Pad with Constant Tensor(s)
};
std::ostream &operator<<(std::ostream &, ConstantPadding);

class Op;

/**
 * This class extends the alias::Graph class to include memory modifying
 * operations: unary, binary, etc.
 *
 * This Graph class uses the PyTorch '_'-suffix notation: a method with an '_'
 * suffix creates at least one alias between an input and an output.
 * */
class Graph {

public:
  Graph() = default;

  /** Insert a constant Tensor into this Graph, with Shape \a shape */
  TensorId constant(const Shape &shape);

  /** Insert a variable Tensor into this Graph, with Shape \a shape */
  TensorId variable(const Shape &shape);

  /**
   * \return The concatenation of Tensors \a ins along axis \a axis. The
   *         output is a new allocation if t is \a AliasType::Outplace,
   *         otherwise it is view-changing.
   * */
  TensorId concat(const TensorIds &ins, AliasType t, uint64_t axis);

  /** A convenience method: concatenation with AliasType::Inplace */
  TensorId concat_(const TensorIds &ins, uint64_t axis);

  /**
   * A generalization of slice and subSample.
   *
   * \param inTensor The Tensor to sample from.
   *
   * \param t Defines how the output Tensor aliases the input Tensor. If \a t
   *          is \a AliasType::Outplace, then the output Tensor is a new
   *          allocation. Otherwise, the output Tensor is a view of the input
   *          Tensor's allocation.
   *
   * \param where The Region defining where to sample.
   *
   * \sa class Region in region.hpp
   * */
  TensorId
  settSample(const TensorId &inTensor, AliasType t, const Region &where);

  /** A convenience method: settSample with AliasType::Inplace */
  TensorId settSample_(const TensorId &id, const Region &w);

  /**
   * Contiguous slice of a Tensor between the bounds \a l and \a u.
   * */
  TensorId slice(const TensorId &, AliasType, const Lower &l, const Upper &u);

  /** A convenience method: slice with AliasType::Inplace */
  TensorId slice_(const TensorId &tIn, const Lower &l, const Upper &u);

  /**
   * Regular subsample of a Tensor, where every \a stride'th slice (of width
   * 1) is selected, along dimension \a dim so for example if \a stride is 1,
   * then the full Tensor is returned.
   * */
  TensorId
  subSample(const TensorId &, AliasType, int64_t stride, uint64_t dim);

  /** A convenience method: subSample with AliasType::Inplace */
  TensorId
  subSample_(const TensorId &tIn, int64_t stride, uint64_t dimension);

  /**
   * Subsample along every dimension simulantaneously.
   *
   * \param strides The striding size in all of the dimensions.
   *
   * strides must be the same size as the rank of the input Tensor, \a tId.
   * */
  TensorId subSample(const TensorId &tId, AliasType, const Strides &strides);

  /** A convenience method: subSample with AliasType::Inplace */
  TensorId subSample_(const TensorId &, const Strides &);

  /** Reverse a Tensor along all dimensions in \a dims. If a dimension is
   * repeated in \a dims, then the reverse is only applied if the dimension
   * appears an odd number of times.
   *
   * \sa Region::reverse
   * */
  TensorId reverse(const TensorId &tIn, AliasType, const Dimensions &dims);

  /** A convenience method: reverse with AliasType::Inplace */
  TensorId reverse_(const TensorId &tIn, const Dimensions &dims);

  /**
   * Expand a Tensor, broadcasting it along singleton dimensions.
   * This is equivalent to numpy.broadcast_to
   * https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html
   *
   * \param id The TensorId of the Tensor to expand
   *
   * \param shape The Shape of the expanded, output Tensor
   *
   * \return The TensorId of the new Tensor
   * */
  TensorId expand(const TensorId &id, AliasType, const Shape &shape);

  /** A convenience method: expand with AliasType::Inplace */
  TensorId expand_(const TensorId &id, const Shape &shape);

  /**
   * Reshape a Tensor.
   *
   * \param id The TensorId of the Tensor to reshape.
   *
   * \param t How the output Tensor aliases the input Tensor.
   *
   * \param shape The Shape of the output Tensor
   *
   * \return The TensorId of the reshaped, output Tensor.
   * */
  TensorId reshape(const TensorId &id, AliasType t, const Shape &shape);

  /** A convenience method: reshape with AliasType::Inplace */
  TensorId reshape_(const TensorId &id, const Shape &shape);

  /** A convenience method: reshape a Tensor to be of rank-1 */
  TensorId flatten(const TensorId &, AliasType);

  /**
   * Copy, or create an alias of a Tensor.
   *
   * This is equivalent to reverse with dims = {}, or reshape with shapes
   * unchanged, or a slice of the entire Tensor.
   * */
  TensorId identity(const TensorId &id, AliasType t);

  /** A convenience method: identity with AliasType::Inplace */
  TensorId identity_(const TensorId &id);

  /**
   * Permute the dimensions of a Tensor.
   *
   * As an example, if the input has shape (3,5,16), and perm is (1,2,0), the
   * output has Shape (5,16,3).
   * */
  TensorId dimShuffle(const TensorId &, AliasType, const Permutation &perm);

  /** A convenience method: dimShuffle with AliasType::Inplace */
  TensorId dimShuffle_(const TensorId &id, const Permutation &perm);

  /**
   * Binary elementwise operation, which use numpy broadcasting rules, see
   * https://numpy.org/doc/stable/user/basics.broadcasting.html
   *
   * \param arg0 The first argument to the binary operation
   *
   * \param arg1 The second argument to the binary operation
   *
   * \param t Defines how the inputs are aliased to the output. The 3
   *          supported cases are:
   *          AliasType::outplace() The output is a new allocation
   *          AliasType::binary0() The output is an alias of \a arg0
   *          AliasType::binary1() The output is an alias of \a arg1
   *          Specifically, AliasType::binary0() denotes an inplace binary
   *          operation where \a arg0 is updated, and the output is an alias
   *          of \a arg0.
   *
   * */
  TensorId binary(const TensorId &arg0, const TensorId &arg1, AliasType t);

  /**
   * Unary elementwise operations
   *
   * \param inTensor The Tensor to perform the elementwise operation on.
   *
   * \param t Defines if the operation is inplace or not. If t is
   *          AliasType::Outplace, the output is a new allocation. Otherwise,
   *          the operation is performed inplace. All inplace operations are
   *          assumed to modify the input.
   *
   * */
  TensorId unary(const TensorId &inTensor, AliasType t);

  /** A convenience method: unary with AliasType::Inplace */
  TensorId unary_(const TensorId &inTensor);

  /**
   * A convenience method, which creates one or multiple allocations and
   * concatenates them around the edges of the input Tensor, \a inTensor. The
   * amount of padding below and above in each dimension is defined by \a
   * lowerPadding and \a upperPadding.
   *
   * \param inTensor The Tensor to pad
   *
   * \param lowerPadding The amount of padding to concatenate at the start of
   *                     each dimension
   *
   * \param upperPadding The amount of padding to concatenate at the end of
   *                     each dimension
   *
   * \param constantPadding This defines whether the allocations which are
   *                        used to pad \a inTensor or constant or variable.
   *
   * \param broadcastPadding This defines if the padding is a single scalar
   *                         value, broadcast to all padding, or if each
   *                         padding element is distinct.
   *
   * The input Tensor \a inTensor will be aliased by the returned output
   * Tensor. So perform an "outplace" pad operation, an the pad must be
   * chained with an call to the identity method.
   * */
  TensorId pad_(const TensorId &inTensor,
                const LowerPadding &lowerPadding,
                const UpperPadding &upperPadding,
                ConstantPadding,
                BroadcastPadding);

  TensorId pad_(const TensorId &inTensor,
                const std::array<std::vector<int64_t>, 2> &lowerAndUpper,
                bool paddingIsParallelWriteable);

  /**
   * A general purpose Op which cannot change its aliasing semantics once
   * constructed.
   *
   * \param ins The Tensor inputs
   *
   * \param outShapes The Shapes of the ouput Tensors
   *
   * \param mapping How inputs and outputs are aliased. Any output can be
   *                aliased to, with or without modification to an input.
   *                There are no view changes supported between inputs and
   *                outputs, so effectively only "unary" and "identity" are
   *                supported.
   *
   * \return The OpId of created Op.
   *
   * This member can be used to represent operations such as convolutions,
   * reductions, etc.
   * */
  OpId multi(const TensorIds &ins,
             const Shapes &outShapes,
             const std::vector<CrossAlias> &mapping);

  OpId noAlias(const TensorIds &ins, const Shapes &outs) {
    return multi(ins, outs, {});
  }

  /**
   * \return All Consumers of the Tensor with TensorId \a id.
   *
   * Recall that a Consumer is defined by 1) an OpId and 2) an InIndex.
   * Specifically the InIndex of a returned Consumer is the index at which
   * this Tensor is consumed.
   *
   * \sa Consumer
   * */
  const Consumers &consumers(const TensorId &id) const;

  /**
   * \return The subset of the Consumers of this Tensor which also modify this
   *         Tensor
   * */
  Consumers modifiers(const TensorId &id) const;

  /** Set the name of the Op with OpId \a id to be \a dbs */
  void setName(OpId, const std::string &dbs);

  /**
   * Set the name of the Op which creates the Tensor with TensorId \a tid, to
   * be \a dbs
   * */
  void setName(const TensorId &tid, const std::string &dbs);

  /** \return The Shape of the Tensor with TensorId \a tid */
  Shape shape(const TensorId &tid) const;

  uint64_t rank_u64(const TensorId &tid) const {
    return shape(tid).rank_u64();
  }

  /**
   * \return The Shapes of the Tensors with TensorIds in \a tids, in the same
   *         order as \a tids.
   * */
  Shapes shapes(const TensorIds &tids) const;

  /**
   * Insert a topological constraint.
   *
   * This constraint ensures that Op with OpId \a before appears before \a
   * after in a scheduling of the Ops in this Graph.
   * */
  void constraint(OpId before, OpId after);

  /**
   * Insert a topological constraint between the creators of Tensors.
   *
   * Specifically, the Op which create \a before must appear before \a after
   * in a linearization of this Graph.
   * */
  void constraint(const TensorId &before, const TensorId &after) {
    constraint(before.opId(), after.opId());
  }

  /**
   * Insert a chain of topological constraints.
   *
   * \a a -> \a b -> \a c -> ...
   *
   * */
  template <class Arg0, class... Args>
  void constraint(Arg0 a, Arg0 b, Args... c) {
    constraint(a, b);
    constraint(b, c...);
  }

  /**
   * Insert topological constraints which ensure that Op with OpId \a before
   * appears before all Ops in \a afters.
   * */
  void constraints(OpId before, const OpIds &afters);

  /**
   * Constraint that all Ops in \a befores appear before \a after in a linear
   * serialization of this Graph.
   * */
  void constraints(const OpIds &befores, OpId after);

  /**
   * Insert multiple constraints
   * */
  void constraints(const Constraints &);

  /**
   * Attempt to make an Op inplace.
   *
   * If the proposed change is accepted, the proposed Op is inplaced and new
   * constraints are inserted between Ops. If the proposed change is rejected,
   * the proposed Op is unchanged, and there are no new constraints inserted.
   *
   * \param proposal The proposed Op to inplace, and the AliasType to try.
   *
   * \param check Whether to disallow inplacing because of writes to non
   *              parallel writeable Tensors.
   *
   * \return The status of the attempt, describing whether or not the change
   *         took place. Possible failure Statuses:
   *
   * Cycle: Sometimes, making an Op inplace results in new constraints between
   *        Ops, to ensure that Tensors which are not modified too early,
   *        trashing memory which is used later. Sometimes, these constraints
   *        result in cycles, in which case the inplacing is rejected.
   *
   * NotParallelWriteable: Sometimes, making an Op inplace results in a Tensor
   *                       which is not parallel writeable beging modified. If
   *                       this happens, and \a check is Yes, the proposal is
   *                       rejected.
   *
   * AlreadyInplace: If the Op is already inplace, the proposal is rejected.
   *
   *
   * \see InplaceResult
   * \see Proposal
   *
   * */
  InplaceStatus tryInplace(const Proposal &proposal,
                           CheckParallelWriteable check);

  /**
   * Attempt to make an Op inplace, but do not perform the final modifications
   * to constraints or this Graph's representation.
   *
   * Only certain changes to Graph are made: no new constraints are inserted.
   * In psuedocode:
   *
   * def tryInplace(.)
   *  result = tryInplacePartial(.)
   *  if (result.valid):
   *    completeInplace(.);
   *
   * There are certain use cases (PopART) where it might still be decided that
   * an inplacing is not desirable, even after tryInplace has confirmed that
   * it is valid. This is why it is useful to expose this sub-routine of
   * tryInplace.
   * */
  InplaceResult tryInplacePartial(const Proposal &proposal,
                                  CheckParallelWriteable);

  /** Perform final Graph modifications */
  void completeInplace(const InplaceResult &inplaceResult);

  /** Revert the changes made in tryInplacePartial, if there were any */
  void backoutInplace(const Proposal &);

  /**
   * Attempt all Proposals in \a proposals, in order
   *
   * \return The InplaceResults of each of the Proposals in \a proposals
   * */
  std::vector<InplaceStatus> tryInplaces(const Proposals &proposals,
                                         CheckParallelWriteable);

  /**
   * Generate Proposals by tying each TensorId in \a ids to
   * AliasType::allInplace
   * */
  static Proposals createProposalsAllInplace(const TensorIds &ids);

  /**
   * Append a string describing this Graph to \a ost
   * */
  void append(std::ostream &ost) const;

  /**
   * \return the AliasType of the Op with OpId \a id
   * */
  AliasType aliasType(OpId id) const;
  AliasType aliasType(TensorId id) const { return aliasType(id.opId()); }

  /**
   * \return The total number of Tensors in this Graph
   * */
  uint64_t nTensors() const;

  /**
   * \return The total number of Ops in this Graph
   * */
  uint64_t nOps() const { return ops.size(); }
  int64_t nOps_i64() const { return static_cast<int64_t>(ops.size()); }

  /**
   * \return All Tensors which are aliased to Tensor \a id
   * */
  TensorIds allAliases(const TensorId &id) const;

  /**
   * \return The number of inputs that the Op \a id has
   * */
  uint64_t nInTensors(OpId id) const;

  /**
   * \return The string description of the Op \a id.
   * */
  std::string typeString(OpId id) const;

private:
  // Strip the OutputIndex of the TensorIds, returning the OpIds -- of the
  // Tensor creators -- of the Tensors in \a tids
  static OpIds getOpIds(const TensorIds &tids);

  template <class T, class... Args>
  OpId createOp(const TensorIds &inIds,
                const Shapes &outShapes,
                AliasType,
                Args... args);

  Op &op(OpId);
  const Op &op(OpId) const;

  // "a minus b" or "a \ b"
  TensorIds difference(TensorIds a, TensorIds b) const;

  // Get a simple edge map, for offloading to a scheduling algorithm.
  using FwdEdges = std::vector<std::vector<decltype(OpId().get())>>;
  FwdEdges getFwdEdges(const Constraints &additional) const;

  alias::Graph &aGraph() { return aGraph_; }
  const alias::Graph &aGraph() const { return aGraph_; }

  void setSchedule(OpIds &&);
  uint64_t scheduleIndex(OpId id) const;

  // Return true if the Constraints constraints are all already satisfied by
  // the current schedule.
  bool satisifedWithoutAnyChange(const Constraints &constraints) const;

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

private:
  // Wrapping a unique_ptr in a class to make the Graph class copyable.
  class UpOp {
  public:
    UpOp();
    UpOp(std::unique_ptr<Op> x);
    UpOp(const UpOp &x);
    UpOp &operator=(const UpOp &x);
    ~UpOp();
    std::unique_ptr<Op> up;
    bool operator==(const UpOp &) const;
  };
  std::vector<UpOp> ops;
  TensorMap tensorMap;

  std::vector<std::string> getOpNames() const;

  // An Op in this Graph maps to 1 or more Nodes in an alias::Graph. For
  // example, an inplace Reshape Op in this Graph maps directly to an Inplace
  // Node in the alias::Graph. An Outplace BinaryOp in this Graph maps to an
  // Allocation in this alias::Graph.
  alias::Graph aGraph_;

  // Schedule and mapping from OpIds to schedule indices.
  std::vector<OpId> sched;
  std::vector<uint64_t> invSched;
  bool scheduleIsValid{false};
};

std::ostream &operator<<(std::ostream &, const Graph &);

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
