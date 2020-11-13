// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_TENSORID_HPP
#define POPRITHMS_MEMORY_INPLACE_TENSORID_HPP

#include <poprithms/memory/inplace/usings.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

/** A Tensor is identified by (OpId, OutIndex) */
class TensorId {
public:
  TensorId() = default;
  TensorId(OpId opId__, OutIndex outIndex__)
      : opId_(opId__), outIndex_(outIndex__) {}

  OpId opId() const { return opId_; }
  OutIndex outIndex() const { return outIndex_; }
  void append(std::ostream &) const;
  std::string str() const;

  bool operator==(const TensorId &rhs) const {
    return getTuple() == rhs.getTuple();
  }
  bool operator!=(const TensorId &rhs) const { return !operator==(rhs); }

  bool operator<(const TensorId &rhs) const {
    return getTuple() < rhs.getTuple();
  }
  bool operator>(const TensorId &rhs) const {
    return getTuple() > rhs.getTuple();
  }

  bool operator<=(const TensorId &rhs) const {
    return getTuple() <= rhs.getTuple();
  }
  bool operator>=(const TensorId &rhs) const {
    return getTuple() >= rhs.getTuple();
  }

  std::tuple<OpId, OutIndex> getTuple() const { return {opId(), outIndex()}; }

private:
  OpId opId_;
  OutIndex outIndex_;
};

std::ostream &operator<<(std::ostream &, const TensorId &);
std::ostream &operator<<(std::ostream &, const TensorIds &);

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
