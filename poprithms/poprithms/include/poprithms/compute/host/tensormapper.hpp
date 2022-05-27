// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMPUTE_HOST_TENSORMAPPER_HPP
#define POPRITHMS_COMPUTE_HOST_TENSORMAPPER_HPP

#include <numeric>
#include <sstream>
#include <variant>

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/memory/nest/region.hpp>

namespace poprithms {
namespace compute {
namespace host {

using ndarray::Dimensions;
using ndarray::Shape;
using Lower     = ndarray::Shape::Lower;
using Upper     = ndarray::Shape::Upper;
using Stride    = ndarray::Stride;
using Dimension = ndarray::Dimension;
using memory::nest::Region;
using util::Permutation;

/**
 * A utility class for transforming a tensor.
 * */
class TensorMapper {
public:
  static Tensor reshape(const Tensor &x, const Shape &s);

  static Tensor expand(const Tensor &x, const Shape &s);

  enum class ReductionType { Sum, Product, Min, Max };
  static Tensor reduce(const Tensor &x,
                       const Shape &s,
                       ReductionType rt = ReductionType::Sum);

  static Tensor settSample(const Tensor &x, const Region &r);

  static Tensor settFillInto(const Tensor &x, const Region &r);

  static Tensor reverse(const Tensor &x, const Dimensions &d);

  static Tensor dimShuffle(const Tensor &x, const Permutation &p);
};

} // namespace host
} // namespace compute
} // namespace poprithms

#endif
