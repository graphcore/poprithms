// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_MULTIOUT_TENSORID_HPP
#define POPRITHMS_COMMON_MULTIOUT_TENSORID_HPP

#include <ostream>
#include <tuple>
#include <vector>

#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/util/hashcombine.hpp>

namespace poprithms {
namespace common {
namespace multiout {

class TensorId;
using TensorIds = std::vector<TensorId>;

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

  static TensorIds flatten(const std::vector<TensorIds> &);

private:
  OpId opId_;
  OutIndex outIndex_;
};
std::ostream &operator<<(std::ostream &, const TensorId &);
std::ostream &operator<<(std::ostream &, const TensorIds &);

} // namespace multiout
} // namespace common
} // namespace poprithms

// To enable hashing of new classes, this is the recommended approach from
// https://en.cppreference.com/w/cpp/utility/hash
namespace std {
template <> struct hash<poprithms::common::multiout::TensorId> {
  std::size_t operator()(
      poprithms::common::multiout::TensorId const &tId) const noexcept {

    using namespace poprithms::common::multiout;
    using namespace poprithms::util;
    size_t seed = 0;
    hash_combine(seed, std::hash<OpId>{}(tId.opId()));
    hash_combine(seed, std::hash<OutIndex>{}(tId.outIndex()));
    return seed;
  }
};
} // namespace std

#endif
