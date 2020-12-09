// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_TENSOR_HPP
#define POPRITHMS_MEMORY_INPLACE_TENSOR_HPP

#include <poprithms/memory/inplace/graph.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

class Tensor;
using Tensors = std::vector<Tensor>;

class Graph;

/**
 * A class to allow for writing "Tensor base code", which is often more
 * succint than can be achiecved with just the Graph class directly. This
 * class does not add any additional functionality. "Tensor-centric" code
 * might be,
 *
 *  <code>
 *    Graph g;
 *    auto r = Tensor::variable(g, {3,3}).flatten().slice({2},{4}).reverse();
 *  </code>
 *
 *  , this in place of the "Graph-centric" equivalent code,
 *
 *  <code>
 *    Graph g;
 *    auto v = g.variable({3,3});
 *    auto f = g.flatten(v);
 *    auto s = g.slice(f, {2}, {4});
 *    auto r = g.reverse();
 *  </code>
 *
 * Warning: the user must ensure that the Graph of a Tensor is not deleted
 * before the final use of a Tensor, as Tensor objects store raw pointers to
 * their Graph.
 * */
class Tensor {

public:
  /** Create a Variable Tensor in a Graph */
  static Tensor variable(Graph &, const Shape &);

  /** Create a Constant Tensor in a Graph */
  static Tensor constant(Graph &, const Shape &);

  Tensor()        = delete;
  Tensor &operator=(const Tensor &) = default;
  Tensor &operator=(Tensor &&) = default;
  Tensor(const Tensor &)       = default;
  Tensor(Tensor &&)            = default;

  bool operator==(const Tensor &rhs) const {
    return id() == rhs.id() && graph_ == rhs.graph_;
  }
  bool operator<(const Tensor &rhs) const {
    return id() < rhs.id() && graph_ < rhs.graph_;
  }

  bool operator!=(const Tensor &rhs) const { return !operator==(rhs); }
  bool operator<=(const Tensor &rhs) const { return !operator>(rhs); }
  bool operator>=(const Tensor &rhs) const { return !operator<(rhs); }
  bool operator>(const Tensor &rhs) const {
    return !operator==(rhs) && !operator<(rhs);
  }

  TensorId id() const { return id_; }
  OpId opId() const { return id().opId(); }

  /** The name of the Graph to which this Tensor belongs */
  std::string graphName() const;

  /** A 1-input Mux on this Tensor. If \a isOpen is true, the returned Tensor
   * is an alias of this Tensor, otherwise it is a new allocation.
   * */
  Tensor mux(bool isOpen) const;
  Tensor openMux() const { return mux(true); }
  Tensor closedMux() const { return mux(false); }

  /** Sample this Tensor. This generalizes slice and subSample. */
  Tensor settSample(const Region &) const;

  /** Slice this Tensor between the bounds #l and #u. */
  Tensor slice(const Lower &l, const Upper &u) const;

  /** Subsample this Tensor along a single dimension */
  Tensor subSample(int64_t stride, uint64_t dimension) const;

  /** Subsample this Tensor along all dimensions. */
  Tensor subSample(const Strides &strides) const;

  /** Reverse a Tensor along all dimensions in #dims. If a dimension is
   * repeated in #dims, then the reverse is applied once for each of the
   * repeats. */
  Tensor reverse(const Dimensions &dims) const;

  Tensor reshape(const Shape &shape) const;

  /** Reshape to be of rank 1 */
  Tensor flatten() const;

  /** Expand a Tensor, broadcasting it along singleton dimensions.
   *  This is equivalent to numpy.broadcast_to
   *  https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html
   *
   * \param shape The Shape of the expanded, output Tensor. */
  Tensor expand(const Shape &shape) const;

  /**  Permute the dimensions of a Tensor. As an example, if this Tensor has
   * Shape (3,5,16), and perm is (1,2,0), the output has Shape (5,16,3). */
  Tensor dimShuffle(const Permutation &perm) const;

  /** Unary elementwise operation which modifies and aliases the input. */
  Tensor unary() const;

  /**
   * A convenience method, which creates one or multiple allocations and
   * concatenates them around the edges of this Tensor. The amount of padding
   * below and above in each dimension is defined by #lowerPadding and
   * #upperPadding.
   *
   * \param lowerPadding The amount of padding to concatenate at the start of
   *                     each dimension
   *
   * \param upperPadding The amount of padding to concatenate at the end of
   *                     each dimension
   *
   * \param constantPadding This defines whether the allocations which are
   *                        used to pad #inTensor are constant or variable.
   *
   * \param broadcastPadding This defines if the padding is a single scalar
   *                         value, broadcast to all padding, or if each
   *                         padding element is distinct.
   *
   * This Tensor will be aliased by the returned output Tensor. */
  Tensor pad(const LowerPadding &,
             const UpperPadding &,
             ConstantPadding,
             BroadcastPadding) const;

  Tensor pad(const std::array<std::vector<int64_t>, 2> &lowerAndUpper,
             bool paddingIsParallelWriteable) const;

  /** The Shape of this Tensor */
  Shape shape() const;
  uint64_t nelms_u64() const { return shape().nelms_u64(); }
  uint64_t rank_u64() const { return shape().rank_u64(); }

  /** \return All Tensors which are aliased to this Tensor. */
  Tensors allAliases() const;

  /** \return The string description of the creator. */
  std::string opTypeString() const;

  /** Set the name of the Op which creates this Tensor to #dbs */
  void setName(const std::string &dbs);

  /**
   * \return All Consumers of this Tensor
   *
   * Recall that a Consumer is defined by 1) an OpId and 2) an InIndex.
   * Specifically the InIndex of a returned Consumer is the index at which
   * this Tensor is consumed.
   *
   * \sa Consumer
   * */
  Consumers consumers() const;

  /** \return The subset of Consumers of this Tensor which modify this Tensor
   */
  Consumers modifiers() const;

  /** Create a constant Tensor in the same Graph as this Tensor */
  Tensor constant(const Shape &) const;

  /** Create a variable Tensor in the same Graph as this Tensor */
  Tensor variable(const Shape &) const;

  /** Queries for the case where this Tensor is the output of a Mux */
  bool muxIsClosed() const;
  bool muxIsOpen() const { return !muxIsClosed(); }

  /** Create a Mux from a non-empty vector of inputs.
   *
   * Recall that a Mux takes N inputs, and creates an output whose Shape is
   * the numpy-style reduction of the inputs. The output is optionally aliased
   * to one of the inputs.
   * */
  static Tensor mux(const Tensors &, InIndex);
  static Tensor mux(const Tensors &);

  /**
   * \return The concatenation of Tensors #ins along axis #axis. The
   *         output is a view of the inputs, there is no new allocation.
   * */
  static Tensor concat(const Tensors &, uint64_t axis);

  /**
   * A general purpose Op which can be used to represent operations such as
   * convolutions, reductions, etc.
   *
   * \param g The Graph into which the Tensors will be inserted. This must
   *          agree with \a ins.
   *
   * \param ins The Tensor inputs
   *
   * \param outShapes The Shapes of the ouput Tensors
   *
   * \param mapping How inputs and outputs are aliased. Any output can be
   *                aliased to, with or without modification, any input.
   *                There are no view changes supported between inputs and
   *                outputs.
   *
   * \return The created Tensors.
   * */

  static Tensors multi(Graph &,
                       const Tensors &ins,
                       const Shapes &outShapes,
                       const CrossAliases &);

  static TensorIds tensorIds(const Tensors &);
  static OpIds opIds(const Tensors &);
  static Tensors tensors(Graph &g, const TensorIds &);
  static Shapes shapes(const Tensors &);

private:
  TensorId id_;
  Graph *graph_;
  Graph &graph() const { return *graph_; }
  Tensors tensors(const TensorIds &) const;

  /** Constructor used by the Graph class factory methods: */
  friend class Graph;
  Tensor(const TensorId &id__, Graph &graph__)
      : id_(id__), graph_(&graph__) {}
};

std::ostream &operator<<(std::ostream &, const Tensor &);
std::ostream &operator<<(std::ostream &, const Tensors &);

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
