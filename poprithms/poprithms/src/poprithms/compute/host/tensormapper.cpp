// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "error.hpp"

#include <poprithms/compute/host/tensormapper.hpp>

namespace poprithms {
namespace compute {
namespace host {

Tensor TensorMapper::reshape(const Tensor &x, const Shape &s) {
  return x.reshape(s);
}

Tensor TensorMapper::expand(const Tensor &x, const Shape &s) {
  return x.expand(s);
}

enum class ReductionType { Sum, Product, Min, Max };
Tensor TensorMapper::reduce(const Tensor &x,
                            const Shape &s,
                            const ReductionType rt) {
  switch (rt) {
  case ReductionType::Product:
    return x.reduceProduct(s);
  case ReductionType::Sum:
    return x.reduceSum(s);
  case ReductionType::Max:
    return x.reduceMax(s);
  case ReductionType::Min:
    return x.reduceMax(s);
  }

  throw poprithms::compute::host::error(
      "Unrecognised ReductionType in TensorMapper");
}

Tensor TensorMapper::settSample(const Tensor &x, const Region &r) {
  return x.gather(r.getOns());
}

Tensor TensorMapper::settFillInto(const Tensor &x, const Region &r) {
  return x.scatterToZero(r.shape(), r.getOns());
}

Tensor TensorMapper::reverse(const Tensor &x, const Dimensions &d) {
  return x.reverse(d.get());
}
Tensor TensorMapper::dimShuffle(const Tensor &x, const Permutation &p) {
  return x.dimShuffle(p);
}

} // namespace host
} // namespace compute
} // namespace poprithms
