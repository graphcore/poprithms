// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMPUTE_HOST_REGIONUTIL_HPP
#define POPRITHMS_COMPUTE_HOST_REGIONUTIL_HPP

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/memory/nest/region.hpp>

namespace poprithms {
namespace compute {
namespace host {

/**
 * Transformations of tensors which involve a nest::Region.
 *
 * Design note: not making these Tensor member functions, as the Tensor class
 * is already large and tries to match the numpy ndarray API where possible.
 * */
class RegionUtil {

public:
  /**
   * \return true if all elements of the tensor #t in the region #r are zero.
   * */
  static bool allZero(const Tensor &t, const memory::nest::Region &r);
};

} // namespace host
} // namespace compute
} // namespace poprithms

#endif
