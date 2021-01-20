// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_ALIAS_GRAPH_HPP
#define POPRITHMS_MEMORY_ALIAS_GRAPH_HPP
#include <array>
#include <map>
#include <set>
#include <vector>

#include <poprithms/memory/alias/tensor.hpp>
#include <poprithms/memory/alias/usings.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/copybyclone.hpp>
#include <poprithms/util/permutation.hpp>

namespace poprithms {
namespace memory {
namespace alias {

class Node;

/// Defines whether the padding is a single, scalar Tensor, broadcast across
/// edges, or if the padding elements are all distinct and don't contain any
/// aliases between each other.
/// \see Graph::pad
enum class BroadcastPadding {
  No = 0, ///< All elements in the padding are distinct allocations
  Yes     ///< All elements in the padding are aliases of a single scalar
};
std::ostream &operator<<(std::ostream &, BroadcastPadding);

/**
 * A directed acyclic graph (DAG) where the nodes represent Tensors, and the
 * edges represent transformations (concats, slices, dimshuffles, etc.).
 * */
class Graph {

public:
  Graph()              = default;
  Graph(const Graph &) = default;
  Graph(Graph &&)      = default;
  Graph &operator=(const Graph &) = default;
  Graph &operator=(Graph &&) = default;

  // defined in source file to support members which are pointers to forward
  // declared Classes (Node).
  ~Graph(); /* = default; */

  /** Insert an allocation Tensor into the Graph. This corresponds to
   * a variable in Poplar: a Tensor which represents memory on some device.
   * The Tensors created with transformations (concat, dimshuffle, etc.) are
   * views into these allocation Tensors.
   *
   * \param shape The Shape of the allocated Tensor.
   *
   * \param color  The Color of the allocation. Colors can be used for
   *               example to distinguish between const and non-const
   *               allocations.
   * */
  TensorId allocate(const Shape &shape, Color color = 0);

  /** Concatenate Tensors along axis `axis'.
   *
   * \return The TensorId of the new Tensor.
   * */
  TensorId concat(const TensorIds &, uint64_t axis);

  /** A generalized concatenation of Tensors.
   *
   * settfill allows for interleaving Tensors. For example, suppose there are
   * 2 input Tensors, Tensor "0" of Shape (2,3) and Tensor "1" of Shape (2,2).
   * There is only 1 way to concatenate these Tensors, along axis 1;
   *
   * 00011
   * 00011.
   *
   * With settfill, the output Tensor could be
   *
   * 01010
   * 01010.
   *
   * \param ids The Tensors which will be combined to form the output Tensor.
   *
   * \param regions The Regions which each of the Tensors will occupy in the
   *                output Tensor. The number of elements in the i'th Region
   *                must be the same as in the i'th Tensor. The Regions must
   *                partition the output Shape.
   * */
  TensorId settfill(const TensorIds &ids, const DisjointRegions &regions);

  /** Permute the dimensions of a Tensor.
   *
   * \return The TensorId of the new Tensor.
   * */
  TensorId dimshuffle(TensorId, const Permutation &);

  /** Sample elements from a Tensor. This is a generalization
   * of slicing and subsampling. See sett.hpp for details.
   *
   * \return The TensorId of the new Tensor.
   * */
  TensorId settsample(TensorId, const Region &);

  /** Reverse a Tensor along the dimensions `dimensions'.
   *
   * \param dimensions The dimensions to reverse. If a dimension appears
   *                   multiple times in this vector, the reversal is
   *                   repeated, so that only an odd number of appearances has
   *                   an overall effect.
   *
   * \return The TensorId of the new Tensor.
   * */
  TensorId reverse(TensorId, const std::vector<uint64_t> &dimensions);

  /** Reshape a Tensor.
   *
   * \return The TensorId of the new Tensor.
   * */
  TensorId reshape(TensorId, const Shape &);

  /**
   * Expand a Tensor, broadcasting it along singleton dimensions. This is
   * equivalent to numpy.broadcast_to.
   *
   * \return The TensorId of the new Tensor.
   * */
  TensorId expand(TensorId, const Shape &);

  /** Create a Tensor in the Graph which is identical to the input Tensor.  */
  TensorId identity(TensorId);

  /** Clone a Tensor.
   *
   * \return The TensorId of the new Tensor. The returned Tensor has
   *         allocation(s) which mirror the input Tensor's, but are distinct.
   *         In poplar-terms, it is always `order-reserving' and corresponds
   *         to PRESERVE_ALIAS.
   * */
  TensorId clone(TensorId);

  /**
   * Pad a Tensor.
   *
   * \param id The Tensor to pad.
   *
   * \param lowerPadding The amount of padding to apply at the start of each
   *                     dimension
   *
   * \param upperPadding The amount of padding to apply at the end of each
   *                     dimension
   *
   * \param padColor The color of the padding. This can be used, for example,
   *                 to distinguish between constant and non-constant padding.
   * */
  TensorId pad(TensorId id,
               const std::vector<uint64_t> &lowerPadding,
               const std::vector<uint64_t> &upperPadding,
               Color padColor,
               BroadcastPadding);

  Tensor tensor(TensorId id) { return {id, this}; }

  /** \return The Shape of a Tensor in this Graph */
  const Shape &shape(TensorId id) const;

  uint64_t rank_u64(TensorId id) const { return shape(id).rank_u64(); }

  /** \return true if the two argument Tensors intersect */
  bool areAliased(TensorId, TensorId) const;

  /** \return true if the two argument Tensors intersect within a specific
   *          allocation Tensor */
  bool areAliased(TensorId id0, TensorId id1, AllocId ida) const;

  /** \return true if not all elements of this Tensor have distinct locations.
   */
  bool containsAliases(TensorId) const;

  /** \return true if any element of the argument Tensor has Color c. Colors
   *          can be used to distinguish between, for example, const and
   *          non-const elements.
   */
  bool containsColor(TensorId, Color c) const;

  /** \return Ids of all Tensors which are aliased to the argument Tensor. */
  TensorIds allAliases(TensorId) const;

  /** \return A vector of TensorIds, where each element contains all aliases
   *          of a Tensor in \a inIds
   *
   * The i'th TensorIds in the returned vector corresponds to the i'th Tensor
   * in \a inIds.
   * */
  std::vector<TensorIds> allAliases(const TensorIds &inIds) const;

  /** \return All Tensor-Tensor aliased. */
  std::vector<TensorIds> allAliases() const;

  std::map<TensorId, std::set<TensorId>> allAliasesMap() const;

  /** If the input map `m' is different to the map returned by
   * allAliasesMap(), then throw an error with a descriptive message */
  void
  confirmAllAliasesMap(const std::map<TensorId, std::set<TensorId>> &m) const;

  /** Make a Tensor an allocation. Example:
   *
   *       bar   out0     .
   *      /    /          .
   *  in0 - id - out1     .
   *  in1 /               .
   *      \               .
   *       foo            .
   *
   *  calling toAllocation(id, myColor) converts the graph to:
   *
   *       bar   out0     .
   *      /    /          .
   *  in0   id - out1     .
   *  in1                 .
   *      \               .
   *       foo            .
   *
   * If `id' is already an allocation, this has no effect other than to
   * possibly change its Color.
   *
   * \param id Then TensorId of the Tensor to convert to an allocation.
   *
   * \param c The Color of the allocation.
   * */
  void toAllocation(TensorId id, Color c);

  /**
   * Modify the Graph, converting the Tensor allocation with TensorId \a
   * allocId into the concatenation of the Tensors in \a inIds, along
   * dimension \a axis.  Example:
   *
   * Before                     After
   * ------                     -----
   * in0                     .    in0
   *        allocId - foo    .        \  allocId - foo
   *                         .        /
   * in1                     .    in1
   *    \                    .       \
   *     bar                 .        bar
   *
   *
   * If the Shape resulting from concatenating in0 and in1 along dimension
   * axis is not the same as the shape of allocId, an error is thrown.
   * */
  void
  allocationToConcat(const TensorIds &inIds, uint64_t axis, TensorId allocId);

  /**
   * Modify the Graph, converting the allocation \a allocId into a Settsample
   *
   * \param inTensor The Tensor which is sampled (sliced, etc)
   *
   * \param r The Region sampled
   *
   * \param allocId The output of the sample, which must be an allocation
   *                before this method is called.
   * */
  void
  allocationToSettsample(TensorId inTensor, const Region &, TensorId allocId);

  /**
   * Modify the Graph, converting the allocation \a allocId into the
   * dimShuffle of Tensor \a inTensor
   * */
  void allocationToDimshuffle(TensorId inTensor,
                              const Permutation &,
                              TensorId allocId);

  /**
   * Modify the Graph, converting the allocation \a allocId into the
   * Reshape of Tensor \a inTensor. If the number of elements of the 2 Tensors
   * are not the same, an error is thrown.
   * */
  void allocationToReshape(TensorId inTensor, TensorId allocId);

  /**
   * Modify the Graph, converting the allocation \a allocId into the
   * expansion of Tensor \a inTensor.
   * */
  void allocationToExpand(TensorId inTensor, TensorId allocId);

  /**
   * Modify the Graph, converting the allocation \a allocId into the
   * reversal of Tensor \a inTensor, along axes \a dimensions.
   * */
  void allocationToReverse(TensorId inTensor,
                           const std::vector<uint64_t> &dimensions,
                           TensorId allocId);

  /** Insert an identity edge into the Graph from `src' to `dst'.
   * Example:
   *
   *  in0 - src - out0       .
   *                         .
   *   in1 - dst - out1      .
   *        /     \          .
   *    in2 - foo  out2      .
   *
   *  calling toIdentity(src, dst), converts the graph to:
   *
   *  in0 - src - out0       .
   *         \               .
   *  in1    dst - out1      .
   *            \            .
   *  in2 - foo  out2        .
   *
   *
   * If `src' and `dst' to not have the same Shape, an error is thrown.
   *
   * */
  void toIdentity(TensorId src, TensorId dst);

  /** \return True if the elements of this Tensor
   *          1) are distinct (no self-aliases)
   *          2) belong to the same allocation (have the same AllocId)
   *          3) form a contiguous set in the flattened allocation.
   *
   * Example:
   *      If this Tensor has shape=(2,3), with elements:
   *                                                      012
   *                                                      345
   *
   *      And it these elements correspond to the elements in an allocation
   *      Tensor of shape=(2,5):
   *                             ..
   *                             21
   *                             30
   *                             54
   *                             ..
   *
   *      Then this tensor is row-major set contiguous. Note that set
   *      contiguity is a weaker condition than (poplar's) contiguity, where
   *      the specific ordering matters. Indeed, the above example is not
   *      contiguous, although if the allocation Tensor was
   *                             ..
   *                             01
   *                             23
   *                             45
   *                             ..
   *
   *      it would be considered contiguous.
   *
   *
   *
   * */
  bool isRowMajorSetContiguous(TensorId) const;

  void reserve(uint64_t nTensors);

  void append(std::ostream &) const;

  /**
   * Append verbose origins information
   * */
  void appendSettwiseOrigins(std::ostream &ost) const {
    appendOrigins(ost, false);
  }
  void appendBitwiseOrigins(std::ostream &ost) const {
    appendOrigins(ost, true);
  }

  std::string verboseString() const;

  uint64_t nTensors() const { return nodes.size(); }

  // Note that the order in which nodes are inserted into the graph must be
  // the same for equality.
  bool operator==(const Graph &rhs) const;
  bool operator!=(const Graph &rhs) const { return !operator==(rhs); }

  enum class Direction { Fwd = 0, Bwd };

  /**
   * A string representation of the transformation resulting in Tensor \a id
   */
  std::string typeString(TensorId id) const;

  /** The inputs of Tensor \a id. These are all of the Tensors of which Tensor
   * \a id is composed. For Tensors created with Reshape, DimShuffle, etc,
   * this is a singleton vector. For Tensors created with Concat, it is all
   * the Tensors concatenated. For allocations, it is empty. */
  const TensorIds &ins(TensorId id) const;

  /** All the Tensors which are composed with Tensor \a id. If "a" is in
   * outs("b") then "b" is in ins("a").  */
  const TensorIds &outs(TensorId id) const;

  /** Return true iff Tensor \a id is an allocation */
  bool allocates(TensorId id) const;

private:
  Node &node(TensorId);
  const Node &node(TensorId) const;

  /**
   * There are two formats for printing the Origins, controlled by \a bitwise.
   *
   * If bitwise = true, the compact Sett representation is used.
   *
   * If bitwise = false, the more verbose and complete representation is used,
   * where for for every dimension, Setts are expanded into  strings of '1'
   * and '0'.
   * */
  void appendOrigins(std::ostream &, bool bitwise) const;

  using UpNode = util::CopyByClone<Node>;
  std::vector<UpNode> nodes;

  // a mutable workspace used for depth-first searches.
  class Workspace {
  public:
    std::vector<bool> wsBool_;
    std::vector<uint64_t> wsUint64_;
    void resize(uint64_t);
    uint64_t size() const { return wsBool_.size(); }
    void clear(const TensorIds &);
    void reserve(uint64_t);
  } mutable wspace;

  template <class T, class... Args>
  TensorId
  createNode(const TensorIds &ins, const Shape &outShape, Args... args);

  template <class T, class... Args>
  std::unique_ptr<T> createNodeWithOutsAndId(const TensorIds &ins,
                                             const TensorIds &outs,
                                             const Shape &outShape,
                                             TensorId id,
                                             Args... args);

  template <class T, class... Args>
  void completeInputlessReplacement(TensorId beingTransformed,
                                    const TensorIds &newIns,
                                    Args... args);

  // post-order depth-wise backwards search for all TensorIds for which f is
  // true.
  template <typename F> TensorIds depthFirstBwd(TensorId id, F &&f) const;

  template <typename F> TensorIds depthFirstFwd(TensorId id, F &&f) const;

  // traverse back collecting all Tensors aliased to id
  TensorIds depthFirstBwdAliases(TensorId id) const;

  TensorIds depthFirstFwdAliases(TensorId id) const;

  // traverse back collecting all Tensors
  TensorIds depthFirstBwdAll(TensorId id) const;

  // set the Origins of Tensor with id `id'
  void setOrigins(TensorId id);

  std::vector<Shape> getShapes(const TensorIds &) const;

  std::vector<std::array<TensorId, 2>>
  createBroadcastPadElements(const Shape &,
                             const std::vector<uint64_t> &lowers,
                             const std::vector<uint64_t> &uppers,
                             Color padColor);

  std::vector<std::array<TensorId, 2>>
  createNonAliasedPadElements(const Shape &,
                              const std::vector<uint64_t> &lowers,
                              const std::vector<uint64_t> &uppers,
                              Color padColor);

  template <Direction D, class F>
  TensorIds depthFirst(TensorId x0, F &&f) const;

  /** A method used when converting allocations to view changes. It tests that
   * \a id is an allocation, and that it has Shape \a expectedShape. If either
   * conditition is not satisified, a clear error is thrown. */
  void assertFromAllocation(TensorId id, const Shape &expectedShape) const;
};

std::ostream &operator<<(std::ostream &, const Graph &);

} // namespace alias
} // namespace memory
} // namespace poprithms

#endif
