// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_CROSS_ALIAS_HPP
#define POPRITHMS_MEMORY_INPLACE_CROSS_ALIAS_HPP

#include <poprithms/memory/inplace/usings.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

/**
 * For an Op which creates an alias between between an input as a specific
 * InIndex and an output at a specific OutIndex, define if the alias is
 * modifying or not.
 * */
struct CrossAlias {
public:
  enum class Modifying { No = 0, Yes };
  CrossAlias(InIndex i_, OutIndex o_, Modifying m_)
      : inIndex_(i_), outIndex_(o_), modifying_(m_) {}

  bool isModifying() const { return modifying_ == Modifying::Yes; }

  InIndex in() const { return inIndex_; }
  OutIndex out() const { return outIndex_; }

  uint64_t in_u64() const { return in().get(); }
  uint64_t out_u64() const { return out().get(); }

  void append(std::ostream &) const;
  bool operator==(const CrossAlias &rhs) const {
    return in() == rhs.in() && out() == rhs.out() &&
           modifying_ == rhs.modifying_;
  }
  bool operator!=(const CrossAlias &rhs) const { return !operator==(rhs); }

private:
  InIndex inIndex_;
  OutIndex outIndex_;
  Modifying modifying_;
};

std::ostream &operator<<(std::ostream &, const CrossAlias &);

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
