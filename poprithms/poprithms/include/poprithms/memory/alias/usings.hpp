// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_ALIAS_ALIASUSINGS_HPP
#define POPRITHMS_MEMORY_ALIAS_ALIASUSINGS_HPP

#include <vector>

#include <poprithms/util/typedinteger.hpp>

namespace poprithms {
namespace memory {
namespace alias {

using InIndex   = poprithms::util::TypedInteger<'i', uint64_t>;
using OutIndex  = poprithms::util::TypedInteger<'o', uint64_t>;
using AllocId   = poprithms::util::TypedInteger<'a', uint32_t>;
using Color     = poprithms::util::TypedInteger<'c', uint32_t>;
using Colors    = std::vector<Color>;
using TensorId  = poprithms::util::TypedInteger<'t', uint32_t>;
using TensorIds = std::vector<TensorId>;

} // namespace alias
} // namespace memory
} // namespace poprithms

#endif
