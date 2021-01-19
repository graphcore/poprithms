// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_MULTIOUT_TENSORID_HPP
#define POPRITHMS_COMMON_MULTIOUT_TENSORID_HPP
#include <ostream>
#include <tuple>
#include <vector>

#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/opid.hpp>

namespace poprithms {
namespace common {
namespace multiout {

/**
 * A Tensor within a Graph is identified by the OpId of the Op which creates
 * it, and the output index where it is created.
 * */
class TensorId {

public:
  TensorId() = default;
  TensorId(OpId opId__, OutIndex outIndex__)
      : opId_(opId__), outIndex_(outIndex__) {}

  /** The Op which creates the Tensor. */
  OpId opId() const { return opId_; }

  /** The output index where this Tensor is created. */
  OutIndex outIndex() const { return outIndex_; }

  void append(std::ostream &) const;
  std::string str() const;

  bool operator==(const TensorId &rhs) const { return tup() == rhs.tup(); }
  bool operator<(const TensorId &rhs) const { return tup() < rhs.tup(); }
  bool operator>(const TensorId &rhs) const { return tup() > rhs.tup(); }
  bool operator!=(const TensorId &rhs) const { return !operator==(rhs); }
  bool operator<=(const TensorId &rhs) const { return !operator>(rhs); }
  bool operator>=(const TensorId &rhs) const { return !operator<(rhs); }

  std::tuple<OpId, OutIndex> tup() const { return {opId(), outIndex()}; }

private:
  OpId opId_;
  OutIndex outIndex_;
};
using TensorIds = std::vector<TensorId>;

std::ostream &operator<<(std::ostream &, const TensorId &);
std::ostream &operator<<(std::ostream &, const TensorIds &);

} // namespace multiout
} // namespace common
} // namespace poprithms

#endif
