// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_CHECK_PARALLEL_WRITEABLE_HPP
#define POPRITHMS_MEMORY_INPLACE_CHECK_PARALLEL_WRITEABLE_HPP
#include <ostream>

namespace poprithms {
namespace memory {
namespace inplace {

/// For some backends, it is required that all Tensors which are modified
/// are also parallel writeable. Being parallel writeable is a strict
/// requirement when targeting Poplar. This enum class defines whether
/// to check that all Tensors which are modified are also parallel
/// writeable. A parallel writeable Tensor is one which contains no Constants,
/// and contains no self-aliases. This is the Poplar defintion of parallel
/// writeability.
/// \see poplar::Tensor
enum class CheckParallelWriteable {
  No =
      0, ///< Allow Tensors which are not parallel writeable to be written to.
  Yes    ///< Only Tensors which are parallel writeable may be be modified,
         ///< ensure that no inplace transformations are applied which do not
         ///< satisfy this.
};
std::ostream &operator<<(std::ostream &, CheckParallelWriteable);

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
