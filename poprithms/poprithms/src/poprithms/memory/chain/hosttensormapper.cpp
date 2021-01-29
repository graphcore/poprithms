// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <memory/chain/hosttensormapper.hpp>
#include <poprithms/memory/chain/error.hpp>

namespace poprithms {
namespace memory {
namespace chain {

compute::host::Tensor
HostTensorMapper::reshape(const compute::host::Tensor &x, const Shape &s) {
  return x.reshape(s);
}

compute::host::Tensor HostTensorMapper::expand(const compute::host::Tensor &x,
                                               const Shape &s) {
  return x.expand(s);
}

enum class ReductionType { Sum, Product, Min, Max };
compute::host::Tensor HostTensorMapper::reduce(const compute::host::Tensor &x,
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

  throw error("Unrecognised ReductionType in HostTensorMapper");
}

compute::host::Tensor
HostTensorMapper::settSample(const compute::host::Tensor &x,
                             const Region &r) {

  return x.gather(r.getOns());
}

compute::host::Tensor
HostTensorMapper::settFillInto(const compute::host::Tensor &x,
                               const Region &r) {
  return x.scatterToZero(r.shape(), r.getOns());
}

compute::host::Tensor
HostTensorMapper::reverse(const compute::host::Tensor &x,
                          const Dimensions &d) {
  return x.reverse(d.get());
}
compute::host::Tensor
HostTensorMapper::dimShuffle(const compute::host::Tensor &x,
                             const Permutation &p) {
  return x.dimShuffle(p);
}

} // namespace chain
} // namespace memory
} // namespace poprithms
