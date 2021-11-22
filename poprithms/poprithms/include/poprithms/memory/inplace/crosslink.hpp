// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_CROSS_LINK_HPP
#define POPRITHMS_MEMORY_INPLACE_CROSS_LINK_HPP

#include <memory>

#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/util/copybyclone.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

using common::multiout::InIndex;
using common::multiout::OutIndex;

/** Desciption of how an input and an output of an Op alias each other */
class CrossLink {

public:
  /** The Tensor at OutIndex #o is a modified alias of the Tensor at InIndex
   * #i. */
  static CrossLink modifies(InIndex i, OutIndex o);

  /** The Tensor at OutIndex #o is an alias of the Tensor at InIndex #i */
  static CrossLink pureIdentityAliases(InIndex i, OutIndex o);

  /**
   * \deprecated { on 19 November 2021. Please use pureIdentityAliases. }
   * */
  static CrossLink pureAliases(InIndex i, OutIndex o) {
    return pureIdentityAliases(i, o);
  }

  bool operator==(const CrossLink &rhs) const { return tup() == rhs.tup(); }
  bool operator!=(const CrossLink &rhs) const { return !operator==(rhs); }

  InIndex in() const { return inIndex_; }
  uint64_t in_u64() const { return in().get(); }

  OutIndex out() const { return outIndex_; }
  uint64_t out_u64() const { return out().get(); }

  bool isModifying() const { return type_ == Type::Modifies; }

  /**
   * The input is exactly the same as the output, without any view-change or
   * modification.
   * */
  bool isPureIdentityAliasing() const {
    return type_ == Type::PureIdentityAliases;
  }
  bool isAliasing() const {
    return isModifying() || isPureIdentityAliasing();
  }

  void append(std::ostream &) const;

private:
  enum class Type { Uses = 0, PureIdentityAliases, Modifies };

  CrossLink(InIndex i_, OutIndex o_, Type t_)
      : inIndex_(i_), outIndex_(o_), type_(t_) {}

  std::tuple<InIndex, OutIndex, Type> tup() const {
    return {in(), out(), type_};
  }

  InIndex inIndex_;
  OutIndex outIndex_;
  Type type_;
};

using CrossLinks = std::vector<CrossLink>;

std::ostream &operator<<(std::ostream &, const CrossLink &);
std::ostream &operator<<(std::ostream &, const CrossLinks &);

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
