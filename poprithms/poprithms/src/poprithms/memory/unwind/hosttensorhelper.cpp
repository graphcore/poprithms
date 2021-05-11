// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <ostream>
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <memory/unwind/ops.hpp>
#include <poprithms/memory/chain/chain.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/memory/unwind/error.hpp>
#include <poprithms/memory/unwind/graph.hpp>
#include <poprithms/memory/unwind/hosttensorhelper.hpp>
#include <poprithms/memory/unwind/path.hpp>
#include <poprithms/memory/unwind/solution.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/stringutil.hpp>
#include <poprithms/util/unisort.hpp>
#include <util/copybyclone_impl.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

compute::host::Tensor HostTensorHelper::get(
    const Solution &soln,
    const TensorId &id,
    const std::map<TensorId, compute::host::Tensor> &sources) {

  using namespace compute;

  if (sources.empty()) {
    std::ostringstream oss;
    oss << "Failed to unwind to host::Tensor " << id
        << ", as no source host Tensors were provided. "
        << "At least one must be provided to determine the numerical type. ";
    throw error(oss.str());
  }

  const auto dtype = sources.cbegin()->second.dtype();
  for (const auto &[id__, t] : sources) {
    (void)id__;
    if (t.dtype() != dtype) {
      std::ostringstream oss;
      oss << "All Tensors in `sources` in this Graph method to "
          << "unwind host Tensors must "
          << "be of the same type. " << t.dtype() << " != " << dtype;
      throw error(oss.str());
    }
  }

  auto out = host::Tensor::zeros(dtype, soln.graph().shape(id));
  for (const auto &p : soln.inwardsPaths(id)) {
    auto found = sources.find(p.src());
    if (found == sources.cend()) {
      std::ostringstream oss;
      oss << "Failed to unwind to host::Tensor, " << id
          << ", as the source of Path " << p << " has source " << p.src()
          << ", which is not present in the provided container of source "
          << "host::Tensors. ";
      throw error(oss.str());
    }
    out = out + p.chain().apply(found->second);
  }
  return out;
}

std::map<TensorId, compute::host::Tensor>
HostTensorHelper::arangeBarriers(const Graph &g) {
  std::map<TensorId, compute::host::Tensor> tensors;
  int64_t start{0};
  for (auto id : g.barriers()) {
    auto end = start + g.nelms(id);
    tensors.insert({id,
                    compute::host::Tensor::arangeInt64(start, end, 1)
                        .reshape(g.shape(id))});
    start = end;
  }
  return tensors;
}

} // namespace unwind
} // namespace memory
} // namespace poprithms
