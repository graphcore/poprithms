// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_CROSS_LINK_HPP
#define POPRITHMS_MEMORY_INPLACE_CROSS_LINK_HPP
#include <memory>

#include <poprithms/memory/inplace/usings.hpp>
#include <poprithms/memory/nest/region.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

/** Abstract base class for mapping DisjointRegions through Ops.
 *
 * For example, a RegsMap describing 2-d max pooling would need to define
 *   fwd(X) = {v : v = {a/2, b/2} for some {a,b} in X}
 *   bwd(Y) = {v : v = {2*a + i, 2*b + j} for some
 *             {a,b} in Y and some {i,b} in {0,1} x {0,1}}
 *
 * Note that for all view-changing operations (Reshape, Slice, DimShuffle,
 * etc) the DisjointRegions class has members implemented already. For
 * example, for DimShuffle with a fixed permutation you would have:
 *   fwd(X) = X.dimSuffle(permutation);
 *   bwd(Y) = Y.dimShuffle(permutation.inverse());
 *
 * \sa DisjointRegion
 * */

class RegsMap {
public:
  virtual DisjointRegions fwd(const DisjointRegions &) const = 0;
  virtual DisjointRegions bwd(const DisjointRegions &) const = 0;
  virtual std::unique_ptr<RegsMap> clone() const             = 0;
  virtual ~RegsMap()                                         = default;

  /** \return true if rhs has same derived class, and same derived properties.
   */
  bool operator==(const RegsMap &rhs) const;
  bool operator!=(const RegsMap &rhs) const { return !operator==(rhs); }

private:
  /**
   * A pure virtual method which derived classes must implement.
   * This function has a precondition that it will only
   * be called when #other is the same type as this instance.
   * */
  virtual bool typeSpecificEqualTo(const RegsMap &other) const = 0;
};

/** Identity mapping of DisjointRegions */
class IdentityRegsMap : public RegsMap {
public:
  DisjointRegions fwd(const DisjointRegions &rs) const final { return rs; }
  DisjointRegions bwd(const DisjointRegions &rs) const final { return rs; }
  virtual std::unique_ptr<RegsMap> clone() const final;

private:
  // All IdentityRegsMaps are equal:
  bool typeSpecificEqualTo(const RegsMap &) const { return true; }
};

/**
 * This class defines a relationship between an input and an output of an Op.
 * There are 3 possibilities:
 *
 * a) the output is a modified alias of the input. This is an equilvanent
 *    relationship to the one defined by #UnaryModifier.
 *
 * b) the output is a pure alias of the input. This is an inplace identity
 *    relationship, equivalent to an open Mux.
 *
 * c) there is no aliasing between the input and output, but the output does
 *    depend on the input. That is, the values of the output depend on a set
 *    of values in the input, where the exact set is defined by a RegsMap.
 *
 * Example:
 *
 * InIndex        OutIndex
 * -------        --------
 *    0 --+
 *        |      +-- 0
 *    1 --+-- Op +
 *        |      +-- 1
 *    2 --+
 *
 *
 * <code>
 *    auto a = modifies(1,0);
 *    auto b = pureAliases(2,1);
 *    auto c = uses(0,1, myMapper);
 * </code>
 *
 * #a Tensor at OutIndex 0 inplace modifies the Tensor at InIndex 1.
 *    \sa UnaryModifier
 *
 * #b Tensor at OutIndex 1 is an alias of Tensor at InIndex 2, without any
 *    modification. In poplar terms, it defines a pure view-change.
 *
 * #c Tensor at OutIndex 1 is not an alias of Tensor at InIndex 0, but its
 *    values are dependant on elements of Tensor at InIndex 0.
 * */

class CrossLink {

public:
  /** The Tensor at OutIndex #o is a modified alias of the Tensor at InIndex
   * #i. */
  static CrossLink modifies(InIndex i, OutIndex o);

  /** The Tensor at OutIndex #o is an alias of the Tensor at InIndex #i */
  static CrossLink pureAliases(InIndex i, OutIndex o);

  /** The Tensor at OutIndex #o is not an alias of the Tensor at InIndex #i,
   * but their values are dependent. */
  static CrossLink
  uses(InIndex i, OutIndex o, std::unique_ptr<RegsMap> regsMap);

  bool operator==(const CrossLink &rhs) const;
  bool operator!=(const CrossLink &rhs) const { return !operator==(rhs); }

  InIndex in() const { return inIndex_; }
  uint64_t in_u64() const { return in().get(); }

  OutIndex out() const { return outIndex_; }
  uint64_t out_u64() const { return out().get(); }

  bool isModifying() const { return type_ == Type::Modifies; }
  bool isPureAliasing() const { return type_ == Type::PureAliases; }
  bool isAliasing() const { return isModifying() || isPureAliasing(); }

  DisjointRegions fwd(const DisjointRegions &in) const {
    return regsMap_.rm_->fwd(in);
  }

  DisjointRegions bwd(const DisjointRegions &out) const {
    return regsMap_.rm_->bwd(out);
  }

  void append(std::ostream &) const;

private:
  enum class Type { Uses = 0, PureAliases, Modifies };

  // Wrapping unique_ptr in a class to make it copyable.
  class WrappedRegsMap {
  public:
    WrappedRegsMap() = default;
    WrappedRegsMap(std::unique_ptr<RegsMap> x) : rm_(std::move(x)) {}

    WrappedRegsMap(const WrappedRegsMap &rhs) : rm_(rhs.rm_->clone()) {}
    WrappedRegsMap(WrappedRegsMap &&rhs);

    WrappedRegsMap &operator=(const WrappedRegsMap &);
    WrappedRegsMap &operator=(WrappedRegsMap &&);

    ~WrappedRegsMap() = default;
    std::unique_ptr<RegsMap> rm_;
  };

  CrossLink(InIndex i_, OutIndex o_, Type t_, std::unique_ptr<RegsMap> r)
      : inIndex_(i_), outIndex_(o_), regsMap_(std::move(r)), type_(t_) {}

  std::tuple<InIndex, OutIndex, Type> tup() const {
    return {in(), out(), type_};
  }

  InIndex inIndex_;
  OutIndex outIndex_;
  WrappedRegsMap regsMap_;
  Type type_;
};

using CrossLinks = std::vector<CrossLink>;

std::ostream &operator<<(std::ostream &, const CrossLink &);
std::ostream &operator<<(std::ostream &, const CrossLinks &);

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
