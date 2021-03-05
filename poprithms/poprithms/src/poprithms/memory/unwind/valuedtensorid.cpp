// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/memory/unwind/valuedtensorid.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

using common::multiout::OpId;
using common::multiout::OutIndex;
using common::multiout::TensorId;

void ValuedTensorId::append(std::ostream &ost) const {
  ost << "(op=" << opId();
  if (tensorId().outIndex() != 0) {
    ost << ",out=" << tensorId().outIndex();
  }
  ost << ",v=" << value() << ')';
}

std::ostream &operator<<(std::ostream &ost, const ValuedTensorId &ps) {
  ps.append(ost);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const ValuedTensorIds &ps) {
  util::append(ost, ps);
  return ost;
}

std::string ValuedTensorId::str() const {
  std::ostringstream oss;
  append(oss);
  return oss.str();
}

void ValuedPair::append(std::ostream &ost) const {
  ost << "(id0=" << id0() << ",id1=" << id1() << ",valPerElm=" << valPerElm()
      << ")";
}

std::ostream &operator<<(std::ostream &ost, const ValuedPair &ps) {
  ps.append(ost);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const ValuedPairs &ps) {
  util::append(ost, ps);
  return ost;
}

} // namespace unwind
} // namespace memory
} // namespace poprithms
