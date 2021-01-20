// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_CHAIN_CHAIN_HPP
#define POPRITHMS_MEMORY_CHAIN_CHAIN_HPP
#include <variant>

#include <poprithms/memory/chain/disjointregionsmapper.hpp>
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

/** A sequence of view-changing operations (Ops).
 *
 * The class has a template method which can be used to apply the Ops in
 * sequence to a Tensor-like class. */
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

  /** The view-changing Ops which can be applied to the end of this Chain.
   * These methods add a new "link" in this Chain and return a reference to
   * this Chain.
   *
   * \sa the class, Region.
   * */
  Chain &reshape(const Shape &);
  Chain &flatten() { return reshape({outShape().nelms()}); }
  Chain &expand(const Shape &);
  Chain &reduce(const Shape &);
  Chain &settSample(const Region &);
  Chain &settSample(const std::vector<nest::Sett> &);
  Chain &slice(const Lower &l, const Upper &u);
  Chain &subSample(Stride s, Dimension d);
  Chain &settFillInto(const Region &);
  Chain &settFillInto(const Lower &l, const Upper &u);
  Chain &settFillInto(Stride s, Dimension d);
  Chain &reverse(const Dimensions &);
  Chain &dimShuffle(const Permutation &);

  /** Append the Chain \a tail to this Chain. \a tail's input Shape must be
   * the same this Op's output Shape. */
  Chain &append(const Chain &tail);

  /** Reverse this Chain. The resulting Chain's input Shape is this Chain's
   * output Shape, reduces are replaced with expands, etc. The mirror of a
   * Chain is its functional inverse. Specifically, if this Chain is f and r =
   * f.mirror(), then f.apply(r.apply(X)) is a subset of X and
   * r.apply(f.apply(X)) is a subset of X.
   * */
  Chain mirror() const;

  /**
   * Template class requirements: `ViewChanger' and `Tensor' must both have
   * methods reshape, expand, reduce, settSample, settFillInto, reverse, and
   * dimShuffle. See for example DisjointRegionsMapper and DisjointRegions.
   *  */
  template <typename ViewChanger, typename Tensor>
  Tensor apply(const Tensor &t_) const {
    static ViewChanger v;
    auto t = t_;
    for (uint64_t opIndex = 0; opIndex < nOps(); ++opIndex) {
      t = get<ViewChanger, Tensor>(opIndex, v, t);
    }
    return t;
  }

  DisjointRegions apply(const DisjointRegions &rIn) const {
    return apply<DisjointRegionsMapper, DisjointRegions>(rIn);
  }

  void append(std::ostream &) const;
  void appendCompact(std::ostream &) const;

  Chain canonicalize() const;

  /** Remove all Ops which are no-ops. */
  Chain removeIdentity() const;

  /** Merge contiguous Ops of the same Type */
  Chain mergeContiguousSameType() const;

  /** \return true of the #n'th Op is an no-Op, such as a Reshape to the same
   *          Shape.  */
  bool isIdentity(uint64_t opIndex) const;

  /** This checks for exact equivalence. That is, #rhs must have exactly the
   * same input Shape, and the exact same sequence of Ops. */
  bool operator==(const Chain &rhs) const;
  bool operator!=(const Chain &rhs) const { return !operator==(rhs); }

  /** Confirm that #rhs is equal to this Chain. If it is not, a descriptive
   * error is thrown. */
  void confirmEqual(const Chain &) const;

  /** Confirm that #rhs is not equal to this Chain. If it is, a descriptive
   * error is thrown. */
  void confirmNotEqual(const Chain &) const;

private:
  /** Unlike most view-changing Graph projects in poprithms, the Chain project
   * does not use polymorphism for the different Op types. This is to
   * facilitate the template method "get" (below) used by the public template
   * method "apply".
   *
   * Instead of using polymorphism, each Op has an enum to describe how it
   * changes the view of a Tensor. */
  enum class Type {
    Reshape = 0,
    SettSample,
    SettFillInto,
    Reverse,
    DimShuffle,
    Expand,
    Reduce,
  };

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
    case Type::Reshape: {
      return g.reshape(t, outShape(opIndex));
    }
    case Type::Expand: {
      return g.expand(t, outShape(opIndex));
    }
    case Type::Reduce: {
      return g.reduce(t, outShape(opIndex));
    }
    case Type::SettSample: {
      return g.settSample(t, region(opIndex));
    }
    case Type::SettFillInto: {
      return g.settFillInto(t, region(opIndex));
    }
    case Type::Reverse: {
      return g.reverse(t, dimensions(opIndex));
    }
    case Type::DimShuffle: {
      return g.dimShuffle(t, permutation(opIndex));
    }
    }
    return t;
  }

  class Op;
  class Attr;
  const Op &op(uint64_t) const;
  Op &op(uint64_t);
  Chain &append(const Op &op);
  template <typename X> Chain &append(Type t, const Shape &o, const X &);
  void append(std::ostream &, uint64_t opIndex) const;
};

std::ostream &operator<<(std::ostream &, const Chain &);

} // namespace chain
} // namespace memory
} // namespace poprithms

#endif
