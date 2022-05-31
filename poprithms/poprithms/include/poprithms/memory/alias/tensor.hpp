// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_ALIAS_TENSOR_HPP
#define POPRITHMS_MEMORY_ALIAS_TENSOR_HPP

#include <string>
#include <vector>

#include <poprithms/memory/alias/usings.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/interval.hpp>
#include <poprithms/util/permutation.hpp>

namespace poprithms {
namespace memory {
namespace alias {

using memory::nest::DisjointRegions;
using memory::nest::Region;
using ndarray::Shape;
using util::Permutation;
using Lower = poprithms::ndarray::Shape::Lower;
using poprithms::ndarray::Dimension;
using Upper     = poprithms::ndarray::Shape::Upper;
using Interval  = poprithms::util::Interval;
using Intervals = std::vector<Interval>;

class Graph;

/**
 *
 * A set based representation of memory addresses of an N-dimensional
 * array.
 *
 * It is useful for answering set based questions such as:
 *
 * 1) Do 2 Tensors intersect?
 *
 * 2) Are all elements in a Tensor unique?
 *
 * 3) Do any/all elements in a Tensor have property X (X = constness)
 *
 *
 * Certain questions cannot be framed in terms of sets. For example, the
 * question
 *
 * Are the elements in a Tensor contiguous?
 *
 * requires an ordering of elements. The similar question,
 *
 * Are the elements in a Tensor setwise contiguous?
 *                              -------
 *
 * can be answered efficiently by this class.
 *
 * How is a Tensor represented? The three important classes in the hierarchy
 * are
 *
 *  - Stripe: 3 integers: "on", "off", and "phase".
 *
 *  - Sett: nested Stripes.
 *
 *  - Region : outer product of Setts.
 *
 * more information on these classes in their respective headers. A Tensor is
 * represented as the union of Regions and the allocation of its elements.
 *
 * In the method comments, we use {a,b} to denote a Tensor with shape (a,b).
 * */
class Tensor;
using Tensors = std::vector<Tensor>;

class Tensor {

public:
  TensorId id() const { return id_; }

  /** \return All Tensors which intersect with this Tensor. */
  Tensors getNonDisjoint() const;

  /**
   * \return All Subtensors in the intervals \a intervals within dimension \a
   *         dim.
   */
  Tensors slices(const Intervals &intervals, uint64_t dim) const;

  /**
   * \return All Subtensors, concatenated using multiple slices (i.e.
   *         intervals). In other words, each sequence of intervals is
   *         concatenated into a single subtensor.
   */
  Tensors slices(const std::vector<Intervals> &intervals, uint64_t dim) const;

  /** \return true if this Tensor intersects with `rhs'. */
  bool intersectsWith(const Tensor &rhs) const;

  /** \return true if not all elements of this Tensor have distinct addresses
   */
  bool containsAliases() const;

  /** \return true if any element of this Tensor has Color c. Colors can be
   *          used to distinguish between, for example, const and non-const
   *          elements (see Graph::allocate).
   */
  bool containsColor(Color c) const;

  /**
   * \return All of the Colors of the allocation(s) which this Tensor is
   *        composed of. The Colors in the returned vector are unique, and in
   *        ascending order.
   * */
  Colors colors() const;

  /**
   * \return Cloned Tensor, which has allocation(s) which mirror this
   *         Tensor's, but are distinct. In poplar terms, it corresponds to
   *         PRESERVE_ALIAS.
   * */
  Tensor clone() const;

  /** Example {10,16}.slice((2,4),(8,7))->{6,3}. */
  Tensor slice(const Lower &, const Upper &) const;

  /** Slice in a single dimension. */
  Tensor slice(uint64_t start, uint64_t end, Dimension) const;

  /** Example {10,16}.flatten()->{160}. */
  Tensor flatten() const;

  /** Example {1,16}.expand(4,5,16)->{4,5,16}. */
  Tensor expand(const Shape &) const;

  /** Example {10,16}.reverse(0)->{10,16}. */
  Tensor reverse(uint64_t dimension) const;

  /** Example {10,16}.reverse((0,1))->{10,16}. */
  Tensor reverse(const std::vector<uint64_t> &dimensions) const;

  /** Example {10,1,16,1}.squeeze()->{10,16}. */
  Tensor squeeze() const;

  /** Example {10,16}.broadcast(3,0)->{30,16}. */
  Tensor broadcast(int64_t N, uint64_t dimension) const;

  /** Examples {10,16}.subsample(5,0)->{2,16}.
   *           {10,16}.subsample(4,0)->{2,16}.
   *           {10,16}.subsample(3,0)->{3,16}. */
  Tensor subsample(int64_t stride, uint64_t dimension) const;

  /** Example {3,5,16}.dimShuffle((1,2,0))->{5,16,3}. */
  Tensor dimShuffle(const Permutation &) const;

  /** Example {4,6}.reshape((3,8))->{3,8}. */
  Tensor reshape(const Shape &) const;

  /**
   * Example: {2,3}.upsample(2,1)->{2,6}.
   *
   * If the tensor in this example with shape {2,3} has values (addresses)
   * ```
   * abc
   * def
   * ```
   * then the upsampled tensor has values
   * ```
   * aabbcc
   * ddeeff
   * ```
   * */
  Tensor upsample(uint64_t scale, uint64_t dim) const;

  /** Take a slice of width 1 of this tensor in dimension-0, then squeeze the
   *  singleton dimension-0 out.
   *  Example {4,5,6}.subscript(1)->{5,6}. */
  Tensor subscript(uint64_t index) const;

  /** Consecutively index the sub-tensor.
   *  Example {2,3,4,5}.index({0,1})->{4,5}. */
  Tensor index(const std::vector<uint64_t> &indices) const;

  /**
   * A generalization of the subsample and slice operators. See region.hpp for
   * details.
   * */
  Tensor settSample(const Region &) const;

  /**
   * Concatenate this Tensor to those in `others', with this Tensor appearing
   * at index "index". Example of how elements are mapped: this = [1,2] others
   * = ([3], [4,5], [6]) index = 1, axis = 0 returns [3,1,2,4,5,6].
   * */
  Tensor concat(const Tensors &others, uint64_t index, uint64_t axis) const;

  /**
   * Concatenate on axis = rank - 1
   * */
  Tensor concatFinalDim(const Tensors &, uint64_t index) const;

  /**
   * Concatenate on axis = 0
   * */
  Tensor concatFirstDim(const Tensors &, uint64_t index) const;

  /**
   * A generalization of concatenation, where the input Tensors map to
   * aribtrary Regions in the output Tensor.
   *
   * \param others The Tensors which, along with this Tensor, will compose
   *               the output Tensor. If there are N Regions in \a regions,
   *               then there must be N - 1 Tensors in \a others.
   *
   * \param thisIndex This Tensor will map to the Region in \a regions at
   *                  index \a thisIndex, of the output Tensor.
   *
   * \see concat
   * \see Graph
   * */
  Tensor settfill(const Tensors &others,
                  uint64_t thisIndex,
                  const DisjointRegions &regions) const;

  /**
   * The Shape of this Tensor.
   * */
  const Shape &shape() const;

  /** Make this Tensor an allocation, disconnecting it from all current
   * inputs. */
  void toAllocation(Color);

  /** Make this Tensor the output of an identity of src, disconnecting it from
   * all current inputs.  */
  void toIdentityFrom(Tensor src);

  // We ignore pgraph, so this is not a strong ordering.
  bool operator<(const Tensor &rhs) const { return id() < rhs.id(); }

  bool operator==(const Tensor &rhs) const {
    return id() == rhs.id() && pgraph == rhs.pgraph;
  }

  int64_t numElements() const { return shape().nelms(); }

  int64_t dim(uint64_t d) const { return shape().dim(d); }

  uint64_t rank_u64() const { return shape().rank_u64(); }

  // see graph.hpp comment for this method
  bool isRowMajorSetContiguous() const;

private:
  friend class Graph;
  TensorId id_;
  Graph *pgraph;
  Tensor(TensorId id, Graph *pg) : id_(id), pgraph(pg) {}
  Tensor() = delete;
};

Tensor concat(const Tensors &, uint64_t axis);
Tensor concat(Tensors &&, uint64_t axis);

/** Generalized concatenation */
Tensor settfill(const Tensors &, const DisjointRegions &);
Tensor settfill(Tensors &&, const DisjointRegions &);

std::ostream &operator<<(std::ostream &, const Tensor &);

} // namespace alias
} // namespace memory
} // namespace poprithms

namespace std {
template <> struct hash<poprithms::memory::alias::Tensor> {
  std::size_t
  operator()(poprithms::memory::alias::Tensor const &tensor) const noexcept {
    return std::hash<decltype(tensor.id())>{}(tensor.id());
  }
};
} // namespace std

#endif
