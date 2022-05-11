// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_HOSTTENSOR_HPP
#define POPRITHMS_COMMON_COMPUTE_HOSTTENSOR_HPP

#include <poprithms/compute/host/tensor.hpp>

namespace poprithms {
namespace common {
namespace compute {

using HostTensor  = poprithms::compute::host::Tensor;
using HostTensors = std::vector<HostTensor>;

/** A host tensor where operator== is defined via numerical equivalence. */
class ComparableHostTensor {
public:
  ComparableHostTensor(const HostTensor &t_) : t(t_) {}
  bool operator==(const ComparableHostTensor &rhs) const {
    return t.numericallyIdenticalTo(rhs.tensor());
  }
  HostTensor tensor() const { return t; }

private:
  HostTensor t;
};

using ComparableHostTensors = std::vector<ComparableHostTensor>;

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
