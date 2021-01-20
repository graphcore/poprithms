// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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
  static CrossLink pureAliases(InIndex i, OutIndex o);

  bool operator==(const CrossLink &rhs) const { return tup() == rhs.tup(); }
  bool operator!=(const CrossLink &rhs) const { return !operator==(rhs); }

  InIndex in() const { return inIndex_; }
  uint64_t in_u64() const { return in().get(); }

  OutIndex out() const { return outIndex_; }
  uint64_t out_u64() const { return out().get(); }

  bool isModifying() const { return type_ == Type::Modifies; }
  bool isPureAliasing() const { return type_ == Type::PureAliases; }
  bool isAliasing() const { return isModifying() || isPureAliasing(); }

  void append(std::ostream &) const;

private:
  enum class Type { Uses = 0, PureAliases, Modifies };

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
