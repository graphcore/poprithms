// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_ALIAS_TENSOR_HPP
#define POPRITHMS_MEMORY_ALIAS_TENSOR_HPP

#include <string>
#include <vector>

#include <poprithms/memory/alias/aliasusings.hpp>
#include <poprithms/ndarray/shape.hpp>

namespace poprithms {
namespace memory {
namespace alias {

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
class Tensor {

public:
  TensorId id() const { return id_; }

  /** \return All Tensors which intersect with this Tensor. */
  std::vector<Tensor> getNonDisjoint() const;

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
   * \return Cloned Tensor, which has allocation(s) which mirror this
   *         Tensor's, but are distinct. In poplar terms, it corresponds to
   *         PRESERVE_ALIAS.
   * */
  Tensor clone() const;

  /** Example {10,16}.slice((2,4),(8,7))->{6,3}. */
  Tensor slice(const Lower &, const Upper &) const;

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

  /** Example {3,5,16}.dimshuffle((1,2,0))->{5,16,3}. */
  Tensor dimshuffle(const Permutation &) const;

  /** Example {4,6}.reshape((3,8))->{3,8}. */
  Tensor reshape(const Shape &) const;

  /**
   * A generalization of the subsample and slice operators. See region.hpp for
   * details.
   * */
  Tensor settsample(const Region &) const;

  /**
   * Concatenate this Tensor to those in `others', with this Tensor appearing
   * at index "index". Example of how elements are mapped: this = [1,2] others
   * = ([3], [4,5], [6]) index = 1, axis = 0 returns [3,1,2,4,5,6].
   * */
  Tensor concat(const std::vector<Tensor> &others,
                uint64_t index,
                uint64_t axis) const;

  /**
   * Concatenate on axis = rank - 1
   * */
  Tensor vstack(const std::vector<Tensor> &ids, uint64_t index) const;

  /**
   * Concatenate on axis = 0
   * */
  Tensor hstack(const std::vector<Tensor> &ids, uint64_t index) const;

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

Tensor concat(const std::vector<Tensor> &, uint64_t axis);
Tensor concat(std::vector<Tensor> &&, uint64_t axis);

std::ostream &operator<<(std::ostream &, const Tensor &);

} // namespace alias
} // namespace memory
} // namespace poprithms

namespace std {
template <> struct hash<poprithms::memory::alias::Tensor> {
  std::size_t operator()(poprithms::memory::alias::Tensor const &tensor) const
      noexcept {
    return std::hash<decltype(tensor.id())>{}(tensor.id());
  }
};
} // namespace std

#endif
