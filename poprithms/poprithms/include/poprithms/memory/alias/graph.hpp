// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_ALIAS_GRAPH_HPP
#define POPRITHMS_MEMORY_ALIAS_GRAPH_HPP
#include <vector>

#include <poprithms/memory/alias/aliasusings.hpp>
#include <poprithms/memory/alias/tensor.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/util/permutation.hpp>
#include <poprithms/util/shape.hpp>

namespace poprithms {
namespace memory {
namespace alias {

class Node;

/**
 * A directed acyclic graph (DAG) where the nodes represent Tensors, and the
 * edges represent transformations (concats, slices, dimshuffles, etc.).
 * */
class Graph {

public:
  Graph() = default;

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
  TensorId concat(const std::vector<TensorId> &, uint64_t axis);

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

  /** Expand a Tensor, broadcasting it along singleton dimensions. This is
   * equivalent to numpy.broadcast_to.
   *
   * \return The TensorId of the new Tensor.
   * */
  TensorId expand(TensorId, const Shape &);

  /** Clone a Tensor.
   *
   * \return The TensorId of the new Tensor. The returned Tensor has
   *         allocation(s) which mirror the input Tensor's, but are distinct.
   *         In poplar-terms, it is always `order-reserving' and corresponds
   *         to PRESERVE_ALIAS.
   * */
  TensorId clone(TensorId);

  Tensor tensor(TensorId id) { return {id, this}; }

  /** \return The Shape of a Tensor in this Graph */
  const Shape &shape(TensorId id) const;

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
  std::vector<TensorId> allAliases(TensorId) const;

  /** \return All Tensor-Tensor aliased. */
  std::vector<std::vector<TensorId>> allAliases() const;

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

  void reserve(uint64_t nTensors) { nodes.reserve(nTensors); }

  void append(std::ostream &) const;

  /**
   * Append verbose origins information
   * */
  void appendVerbose(std::ostream &) const;
  std::string verboseString() const;

  uint64_t nTensors() const { return nodes.size(); }

  // Note that the order in which nodes are inserted into the graph must be
  // the same for equality.
  bool operator==(const Graph &rhs) const { return nodes == rhs.nodes; }
  bool operator!=(const Graph &rhs) const { return !operator==(rhs); }

private:
  Node &node(TensorId);
  const Node &node(TensorId) const;

  // A wrapper around a unique pointer to make it copyable.
  template <class T> class Up {
  public:
    Up();
    Up(std::unique_ptr<T> x);
    Up(const Up &x);
    Up &operator=(const Up &x);
    ~Up();
    std::unique_ptr<T> up;
    bool operator==(const Up<T> &) const;
  };
  using UpNode = Up<Node>;
  std::vector<UpNode> nodes;

  // a mutable workspace used for depth-first searches.
  class Workspace {
  public:
    std::vector<bool> wsBool_;
    std::vector<uint64_t> wsUint64_;
    void resize(uint64_t);
    uint64_t size() const { return wsBool_.size(); }
    void clear(const std::vector<TensorId> &);
    void reserve(uint64_t);
  } mutable wspace;

  // Recall that the Origins of a Tensor contains all the information about
  // the allocations/memory which the Tensor aliases.
  //
  // The Origins of Tensors are only updated when a user requests
  // alias-specific information about them. For example, suppose the user
  // calls:
  //
  // Graph g;
  // auto t0 = g.tensor(g0.allocate({2,3})).slice({0,0},{2,2}).flatten();
  //
  // At this point in the code, the Origins of all Tensors in the graph are
  // unset. If now the user calls
  //
  // auto foo = t0.containsAliases();
  //
  // The Origins of t0 is required, and so a computation is run to update it
  // and all other Origins necessary.
  //
  // Definition of a "stale Tensor": A Tensor which has not had its Origins
  // set, or has had its Origins set but they might have changed since last
  // set.
  //
  // All Tensors which have origins which might need to be updated.
  mutable std::vector<uint64_t> stale_;

  bool isStale(TensorId id) const;

  void makeKnownNewStale(TensorId id) const { stale_.push_back(id.get()); }

  void ensureStale(TensorId id) const {
    if (!isStale(id)) {
      makeKnownNewStale(id);
    }
  }

  size_t nStale() const { return stale_.size(); }

  // Ensure that the origins of a Tensor are up-to-date.
  void updateOrigins(TensorId) const;

  // Ensure that the origins of all Tensors are up-to-date.
  void updateOrigins() const;

  template <class T, class... Args>
  TensorId createNode(const std::vector<TensorId> &ins,
                      const Shape &outShape,
                      Args... args);

  // post-order depth-wise backwards search for all TensorIds for which f is
  // true.
  template <typename F>
  std::vector<TensorId> depthFirstBack(TensorId id, F &&f) const;

  // traverse backwards collecting all stale Tensors
  std::vector<TensorId> depthFirstBackStale(TensorId id) const;

  // traverse back collecting all Tensors aliased to id
  std::vector<TensorId> depthFirstBackAliases(TensorId id) const;

  // traverse back collecting all Tensors
  std::vector<TensorId> depthFirstBackAll(TensorId id) const;

  // traverse forwards til either stale or do not intersect
  void makeStaleForwardAliased(TensorId id) const;

  std::vector<Shape> getShapes(const std::vector<TensorId> &) const;
};

std::ostream &operator<<(std::ostream &, const Graph &);

} // namespace alias
} // namespace memory
} // namespace poprithms

#endif
