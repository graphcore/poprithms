// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_CROSS_ALIAS_HPP
#define POPRITHMS_MEMORY_INPLACE_CROSS_ALIAS_HPP

#include <poprithms/memory/inplace/usings.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

/**
 * This struct defines, for an Op which creates an alias between an input at a
 * specific InIndex, and an output at a specific OutIndex, if the alias
 * between the input and output Tensors is modifying or not.
 *
 * Example: CrossAlias(InIndex=2, OutIndex=1, Modifying=False) defines an
 * alias between the input Tensor at InIndex 2 and the output Tensor at
 * OutIndex 1, without any modifcation. In poplar terms, it defines a pure
 * view-change.
 * */
struct CrossAlias {
public:
  enum class Modifying { No = 0, Yes };
  CrossAlias(InIndex i_, OutIndex o_, Modifying m_)
      : inIndex_(i_), outIndex_(o_), modifying_(m_) {}

  bool operator==(const CrossAlias &rhs) const { return tup() == rhs.tup(); }
  bool operator!=(const CrossAlias &rhs) const { return !operator==(rhs); }
  bool operator<(const CrossAlias &rhs) const { return tup() < rhs.tup(); }

  InIndex in() const { return inIndex_; }
  uint64_t in_u64() const { return in().get(); }

  OutIndex out() const { return outIndex_; }
  uint64_t out_u64() const { return out().get(); }

  bool isModifying() const { return modifying_ == Modifying::Yes; }

  void append(std::ostream &) const;

private:
  std::tuple<InIndex, OutIndex, Modifying> tup() const {
    return {in(), out(), modifying_};
  }
  InIndex inIndex_;
  OutIndex outIndex_;
  Modifying modifying_;
};

using CrossAliases = std::vector<CrossAlias>;

std::ostream &operator<<(std::ostream &, const CrossAlias &);
std::ostream &operator<<(std::ostream &, const CrossAliases &);

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
