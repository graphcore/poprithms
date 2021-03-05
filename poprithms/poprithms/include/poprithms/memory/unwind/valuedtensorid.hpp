// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_UNWIND_VALUEDTENSORID_HPP
#define POPRITHMS_MEMORY_UNWIND_VALUEDTENSORID_HPP

#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

using common::multiout::OpId;
using common::multiout::OutIndex;
using common::multiout::TensorId;

/** A TensorId and a double value. */
struct ValuedTensorId {
public:
  ValuedTensorId(const TensorId &tensorId, double v)
      : tId_(tensorId), value_(v) {}

  TensorId tensorId() const { return tId_; }
  OpId opId() const { return tensorId().opId(); }
  double value() const { return value_; }
  void setValue(double d) { value_ = d; };

  bool operator==(const ValuedTensorId &r) const { return tup() == r.tup(); }
  bool operator<(const ValuedTensorId &r) const { return tup() < r.tup(); }
  bool operator>(const ValuedTensorId &r) const { return tup() > r.tup(); }
  bool operator!=(const ValuedTensorId &r) const { return !operator==(r); }
  bool operator<=(const ValuedTensorId &rhs) const { return !operator>(rhs); }
  bool operator>=(const ValuedTensorId &rhs) const { return !operator<(rhs); }

  std::tuple<TensorId, double> tup() const { return {tId_, value_}; }

  std::string str() const;
  void append(std::ostream &) const;

private:
  TensorId tId_;
  double value_;
};
using ValuedTensorIds = std::vector<ValuedTensorId>;
std::ostream &operator<<(std::ostream &, const ValuedTensorId &);
std::ostream &operator<<(std::ostream &, const ValuedTensorIds &);

/** Two TensorIds and an (elementwise) value */
struct ValuedPair {

public:
  ValuedPair(const TensorId &id0, const TensorId &id1, double v)
      : id0_(id0), id1_(id1), valPerElm_(v) {}

  TensorId id0() const { return id0_; }
  TensorId id1() const { return id1_; }
  double valPerElm() const { return valPerElm_; }

  bool operator==(const ValuedPair &r) const { return tup() == r.tup(); }
  bool operator<(const ValuedPair &r) const { return tup() < r.tup(); }
  bool operator>(const ValuedPair &r) const { return tup() > r.tup(); }
  bool operator!=(const ValuedPair &r) const { return !operator==(r); }
  bool operator<=(const ValuedPair &rhs) const { return !operator>(rhs); }
  bool operator>=(const ValuedPair &rhs) const { return !operator<(rhs); }

  std::tuple<double, TensorId, TensorId> tup() const {
    return {valPerElm_, id0_, id1_};
  }

  std::string str() const;
  void append(std::ostream &) const;

private:
  TensorId id0_;
  TensorId id1_;
  double valPerElm_;
};

using ValuedPairs = std::vector<ValuedPair>;
std::ostream &operator<<(std::ostream &, const ValuedPair &);
std::ostream &operator<<(std::ostream &, const ValuedPairs &);

} // namespace unwind
} // namespace memory
} // namespace poprithms

#endif
