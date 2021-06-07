// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_MULTIOUT_OPTRAVERSAL_HPP
#define POPRITHMS_COMMON_MULTIOUT_OPTRAVERSAL_HPP

#include <ostream>
#include <tuple>
#include <vector>

#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/opid.hpp>

namespace poprithms {
namespace common {
namespace multiout {

/**
 * An (InIndex, OpId, OutIndex) triplet, describing a path through an Op.
 * */

class OpTraversal {

public:
  OpTraversal() = delete;
  OpTraversal(InIndex i, OpId op, OutIndex o)
      : inIndex_(i), opId_(op), outIndex_(o) {}

  /**
   * The index of entry into the Op.
   * */
  InIndex inIndex() const { return inIndex_; }

  /**
   * The Op being traversed.
   * */
  OpId opId() const { return opId_; }

  /**
   * The index of exit from the Op.
   * */
  OutIndex outIndex() const { return outIndex_; }

  void append(std::ostream &) const;
  std::string str() const;

  bool operator==(const OpTraversal &rhs) const { return tup() == rhs.tup(); }
  bool operator!=(const OpTraversal &rhs) const { return tup() != rhs.tup(); }
  bool operator<(const OpTraversal &rhs) const { return tup() < rhs.tup(); }
  bool operator>(const OpTraversal &rhs) const { return tup() > rhs.tup(); }
  bool operator<=(const OpTraversal &rhs) const { return tup() <= rhs.tup(); }
  bool operator>=(const OpTraversal &rhs) const { return tup() >= rhs.tup(); }

  std::tuple<OpId, InIndex, OutIndex> tup() const {
    return {opId(), inIndex(), outIndex()};
  }

private:
  InIndex inIndex_;
  OpId opId_;
  OutIndex outIndex_;
};

using OpTraversals = std::vector<OpTraversal>;

std::ostream &operator<<(std::ostream &, const OpTraversal &);
std::ostream &operator<<(std::ostream &, const OpTraversals &);

} // namespace multiout
} // namespace common
} // namespace poprithms

#endif
