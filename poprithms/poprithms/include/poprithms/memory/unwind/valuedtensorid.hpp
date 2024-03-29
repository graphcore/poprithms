// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_UNWIND_VALUEDTENSORID_HPP
#define POPRITHMS_MEMORY_UNWIND_VALUEDTENSORID_HPP

#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/util/valuedtuple.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

using common::multiout::OpId;
using common::multiout::OutIndex;
using common::multiout::TensorId;

/** A TensorId and a double. */
struct ValuedTensorId
    : public util::ValuedTuple<std::tuple<TensorId, double>> {
public:
  ValuedTensorId(const TensorId &tensorId, double v)
      : ValuedTuple({tensorId, v}) {}
  TensorId tensorId() const { return get<0, TensorId>(); }
  OpId opId() const { return tensorId().opId(); }
  double value() const { return get<1, double>(); }
  void setValue(double d) { setVal<1, double>(d); }
  std::string str() const;
  void append(std::ostream &) const;
};

using ValuedTensorIds = std::vector<ValuedTensorId>;
std::ostream &operator<<(std::ostream &, const ValuedTensorId &);
std::ostream &operator<<(std::ostream &, const ValuedTensorIds &);

/**
 * Two TensorIds and a double. The double is the value of "attraction" between
 * corresponding elements in the 2 tensors.
 * */
struct ValuedPair
    : public util::ValuedTuple<std::tuple<double, TensorId, TensorId>> {

public:
  ValuedPair(const TensorId &id0, const TensorId &id1, double v)
      : ValuedTuple({v, id0, id1}) {}
  TensorId id0() const { return get<1, TensorId>(); }
  TensorId id1() const { return get<2, TensorId>(); }
  double valPerElm() const { return get<0, double>(); }
  std::string str() const;
  void append(std::ostream &) const;
};

using ValuedPairs = std::vector<ValuedPair>;

/**
 * An extension to the ValuedPair struct, which additionally has a uint64_t
 * value. The uint64_t acts as a tie-breaker in comparisons when the doubles
 * are the same in the 2 objects. For the Solution class, the uint64_t is the
 * longest path to a terminal node (see the Solution class for more
 * info).
 * */
struct ExtendedValuedPair
    : public util::ValuedTuple<
          std::tuple<double, uint64_t, TensorId, TensorId>> {

public:
  ExtendedValuedPair(const TensorId &id0,
                     const TensorId &id1,
                     double v,
                     uint64_t lengthToEnd)
      : ValuedTuple({v, lengthToEnd, id0, id1}) {}

  TensorId id0() const { return get<2, TensorId>(); }
  TensorId id1() const { return get<3, TensorId>(); }
  uint64_t lengthToEnd() const { return get<1, uint64_t>(); }
  double valPerElm() const { return get<0, double>(); }
  std::string str() const;
  void append(std::ostream &) const;
};

using ExtendedValuedPairs = std::vector<ExtendedValuedPair>;

std::ostream &operator<<(std::ostream &, const ValuedPair &);
std::ostream &operator<<(std::ostream &, const ValuedPairs &);

} // namespace unwind
} // namespace memory
} // namespace poprithms

#endif
