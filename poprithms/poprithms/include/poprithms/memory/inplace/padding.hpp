// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_PADDING_HPP
#define POPRITHMS_MEMORY_INPLACE_PADDING_HPP

#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/util/typedinteger.hpp>
#include <poprithms/util/typedvector.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

using LowerPadding =
    util::TypedVector<uint64_t, 'L', 'o', 'w', 'P', 'a', 'd'>;

using UpperPadding =
    util::TypedVector<uint64_t, 'U', 'p', 'p', 'P', 'a', 'd'>;

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
