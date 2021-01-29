// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_CHAIN_HOSTTENSORMAPPER_HPP
#define POPRITHMS_MEMORY_CHAIN_HOSTTENSORMAPPER_HPP
#include <numeric>
#include <sstream>
#include <variant>

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/memory/nest/region.hpp>

namespace poprithms {
namespace memory {
namespace chain {

using ndarray::Dimensions;
using ndarray::Shape;
using Lower     = ndarray::Shape::Lower;
using Upper     = ndarray::Shape::Upper;
using Stride    = ndarray::Stride;
using Dimension = ndarray::Dimension;
using nest::Region;
using util::Permutation;

/**
 * Class to help map a compute::host::Tensor through the links of a
 * memory::chain::Chain.
 * */
class HostTensorMapper {
public:
  static compute::host::Tensor reshape(const compute::host::Tensor &x,
                                       const Shape &s);

  static compute::host::Tensor expand(const compute::host::Tensor &x,
                                      const Shape &s);

  enum class ReductionType { Sum, Product, Min, Max };
  static compute::host::Tensor reduce(const compute::host::Tensor &x,
                                      const Shape &s,
                                      ReductionType rt = ReductionType::Sum);

  static compute::host::Tensor settSample(const compute::host::Tensor &x,
                                          const Region &r);

  static compute::host::Tensor settFillInto(const compute::host::Tensor &x,
                                            const Region &r);

  static compute::host::Tensor reverse(const compute::host::Tensor &x,
                                       const Dimensions &d);

  static compute::host::Tensor dimShuffle(const compute::host::Tensor &x,
                                          const Permutation &p);
};

} // namespace chain
} // namespace memory
} // namespace poprithms

#endif
