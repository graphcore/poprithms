// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_CHAIN_CHAIN_HPP
#define POPRITHMS_MEMORY_CHAIN_CHAIN_HPP

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/memory/chain/type.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/copybyclone.hpp>
#include <poprithms/util/permutation.hpp>

namespace poprithms {
namespace memory {
namespace chain {

using ndarray::Dimensions;
using ndarray::Shape;
using Lower     = ndarray::Shape::Lower;
using Upper     = ndarray::Shape::Upper;
using Stride    = ndarray::Stride;
using Dimension = ndarray::Dimension;
using nest::DisjointRegions;
using nest::Region;
using util::Permutation;

class Op;

/**
 * A Chain is a sequence of operations (Ops). Each Op has 1 output Shape, and
 * 1 input Shape which is the output of the preceding Op. A Chain also has an
 * input, which is the input to the first Op.
 *
 * The class has a template method which can be used to apply the Ops in
 * sequence to a Tensor-like class.
 *
 * Chains can be canonicalized. For more information see Chain.md
 * */
class Chain {

public:
  Chain() = delete;

  /** Construct an empty Chain with input Shape \a inputShape */
  Chain(const Shape &inputShape);

  Chain(const Chain &);
  Chain(Chain &&);

  Chain &operator=(const Chain &);
  Chain &operator=(Chain &&);

  ~Chain();

  /** \return The number of view-changing Ops in this Chain. */
  uint64_t nOps() const;

  /** The Shape of the input Tensor to the \a n'th Op in this Chain. */
  Shape inShape(uint64_t n) const;

  /** The Shape of the input Tensor to the \a n'th Op in this Chain. */
  Shape outShape(uint64_t n) const;

  /** The Shape of the input Tensor to the 0'th Op in this Chain. */
  Shape inShape() const;

  /** The Shape of the output Tensor of the final Op in this Chain. */
  Shape outShape() const;

  /**
   * The view-changing Ops which can be applied to the end of this Chain.
   * These methods add a new "link" in this Chain and return a reference to
   * this Chain.
   *
   * \sa the class, Region.
   * */

  void reshape(const Shape &);
  void flatten() { return reshape({outShape().nelms()}); }
  void squeeze() { return reshape({outShape().squeeze()}); }
  void expand(const Shape &);
  void reduce(const Shape &);
  void settSample(const Region &);
  void settSample(const std::vector<nest::Sett> &);
  void slice(const Lower &l, const Upper &u);
  void subSample(Stride s, Dimension d);
  void settFillInto(const Region &);
  void settFillInto(const Lower &l, const Upper &u);
  void settFillInto(Stride s, Dimension d);
  void reverse(const Dimensions &);
  void reverse(Dimension);
  void dimShuffle(const Permutation &);

  /**
   * Apply settSample(window) and then settFillInto(window). This composite
   * operation blocks out all "values" outside of \a window.
   * */
  void mask(const Region &window);

  /** Append the Chain \a tail to this Chain. \a tail's input Shape must be
   * the same this Op's output Shape. */
  void append(const Chain &tail);

  /** Reverse this Chain. The resulting Chain's input Shape is this Chain's
   * output Shape, reduces are replaced with expands, etc. The mirror of a
   * Chain is its functional inverse. Specifically, if this Chain is f and r =
   * f.mirror(), then f.apply(r.apply(X)) is a subset of X and
   * r.apply(f.apply(X)) is a subset of X.
   * */
  Chain mirror() const;

  /**
   * Template class requirements: `ViewChanger' and 'View' must both have
   * methods reshape, expand, reduce, settSample, settFillInto, reverse, and
   * dimShuffle. Two example uses cases are
   *
   *    ViewChanger              View
   *    -----------------------+--------------------------+
   * 1) DisjointRegionsMapper  | DisjointRegions          |
   * 2) HostTensorMapper       | compute::host::Tensor    |
   *    -----------------------+--------------------------+
   *
   * \tparam view The View to apply this Chain to.
   *
   * \tparam nOpsToApply Apply the first #nOpsToApply Ops in this chain to
   *                     #view.
   *  */
  template <typename ViewChanger, typename View>
  View apply(const View &view, uint64_t nOpsToApply) const {
    const ViewChanger viewChanger;
    auto outView = view;
    for (uint64_t opIndex = 0; opIndex < nOpsToApply; ++opIndex) {
      outView = get<ViewChanger, View>(opIndex, viewChanger, outView);
    }
    return outView;
  }

  /**
   * Apply all Ops in this Chain to a View.
   * */
  template <typename ViewChanger, typename View>
  View apply(const View &view) const {
    return apply<ViewChanger, View>(view, nOps());
  }

  /**
   * A method to sequentially apply each link in this Chain to
   * DijsointRegions. It is an instantiation of the template `apply` method.
   * */
  DisjointRegions apply(const DisjointRegions &rIn) const;
  DisjointRegions apply(const DisjointRegions &rIn,
                        uint64_t nOpsToApply) const;

  /**
   * A method to sequentially apply each link in this Chain to a
   * compute::host::Tensor. It is an instantiation of the template `apply`
   * method.
   * */
  compute::host::Tensor apply(const compute::host::Tensor &) const;

  void append(std::ostream &) const;
  void appendCompact(std::ostream &) const;

  /** Perform a sequence of passes on this Chain to simplify and canonicalize
   *  it. These passes include:
   *
   *  - remove no-op operators, such as DimShuffle with the Identity
   *    Permutation.
   *
   *  - merge contiguous operations of the same type. For example,
   *    DimShuffle(perm0) followed by DimShuffle(perm1) becomes
   *    DimShuffle(perm0.mul(perm1))
   *
   *  - remove SettFillInto(r0) followed by SettSample(r0), as these combined
   *    make a no-op.
   *
   *  - try and order the operations in alphabetical order by bubble sorting.
   *    When 2 contiguous operations are swapped, they may change slightly to
   *    preserve the Chain's behaviour.
   *
   * These passes are repeated until the Chain is unchanged.
   *
   *  For more information see Chain.md
   **/
  void canonicalize();

  Chain canonicalized() const;

  /**
   * Check for exact equivalence. That is, #rhs must have exactly the
   * same input Shape, and the exact same sequence of Ops.
   * */
  bool operator==(const Chain &rhs) const;
  bool operator!=(const Chain &rhs) const { return !operator==(rhs); }

  /** Confirm that #rhs is equal to this Chain. If it is not, a descriptive
   * error is thrown. */
  void confirmEqual(const Chain &) const;

  /** Confirm that #rhs is not equal to this Chain. If it is, a descriptive
   * error is thrown. */
  void confirmNotEqual(const Chain &) const;

private:
  /** This method is used in canonicalization. It tries to merge or remove the
   * final 2 Ops in this Chain. */
  bool tryMergeLastTwo();

  /** \return true if the #n'th Op is a no-Op, such as a Reshape to the same
   *          Shape.  */
  bool isIdentity(uint64_t n) const;

  /** This method should only be called by SettSample and SettFillInto Ops. */
  Region region(uint64_t) const;

  /** This method should only be called by the DimShuffle Op. */
  Permutation permutation(uint64_t) const;

  /** This method should only be called by the Reverse Op. */
  Dimensions dimensions(uint64_t) const;

  /** Return the Type of the #n'th Op in this Chain. */
  Type type(uint64_t n) const;

  class Ops;
  util::CopyByClone<Ops> ops_;

  Shape inShape_;

  template <typename G, typename T>
  T get(uint64_t opIndex, const G &g, const T &t) const {
    switch (type(opIndex)) {
    case Type::DimShuffle: {
      return g.dimShuffle(t, permutation(opIndex));
    }
    case Type::Expand: {
      return g.expand(t, outShape(opIndex));
    }
    case Type::Reduce: {
      return g.reduce(t, outShape(opIndex));
    }
    case Type::Reverse: {
      return g.reverse(t, dimensions(opIndex));
    }
    case Type::Reshape: {
      return g.reshape(t, outShape(opIndex));
    }
    case Type::SettFillInto: {
      return g.settFillInto(t, region(opIndex));
    }
    case Type::SettSample: {
      return g.settSample(t, region(opIndex));
    }
    }
    return t;
  }

  Type backType() const { return type(nOps() - 1); }

  /**
   * Consider \a x0 and \a x1, contiguous Ops in a Chain, [a b c x0 x1 d ]
   *
   * This method attempts to to swap the x0 and x1, while ensuring the
   * behaviour of the Chain is unchanged.
   *
   * \param i: The index of the second (x1) Op in the pair to try and swap.
   *
   * \return true if the swap was performed, so that the types of x0 and x1
   *         are swapped, and possibly the Attrs of them change. false if x0
   *         and x1 are unchanged. The swap is performed if (1) x1 < x0
   *         lexicographically an (2) the swap can be performed while
   *         guaranteeing the behaviour of the Chin is unchanged.
   * */
  bool tryBubbleBack(uint64_t i);

private:
  const Op &op(uint64_t) const;
  Op &op(uint64_t);
  void popBack();
  void append(const Op &op);
  template <typename X> void append(Type t, const Shape &o, const X &);
  void append(std::ostream &, uint64_t opIndex) const;
};

using Chains = std::vector<Chain>;

std::ostream &operator<<(std::ostream &, const Chain &);

} // namespace chain
} // namespace memory
} // namespace poprithms

#endif
