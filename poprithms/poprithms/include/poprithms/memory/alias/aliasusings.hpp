// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_ALIAS_ALIASUSINGS_HPP
#define POPRITHMS_MEMORY_ALIAS_ALIASUSINGS_HPP

#include <vector>

#include <poprithms/util/typedinteger.hpp>

namespace poprithms {
namespace util {
class Shape;
class Permutation;
} // namespace util
} // namespace poprithms

namespace poprithms {
namespace memory {
namespace nest {
class Region;
class DisjointRegions;
} // namespace nest
} // namespace memory
} // namespace poprithms

namespace poprithms {
namespace memory {
namespace alias {

using InIndex  = poprithms::util::TypedInteger<'i', uint64_t>;
using OutIndex = poprithms::util::TypedInteger<'o', uint64_t>;
using TensorId = poprithms::util::TypedInteger<'t', uint32_t>;
using AllocId  = poprithms::util::TypedInteger<'a', uint32_t>;
using Color    = poprithms::util::TypedInteger<'c', uint32_t>;

using Shape           = poprithms::util::Shape;
using Permutation     = poprithms::util::Permutation;
using Region          = poprithms::memory::nest::Region;
using DisjointRegions = poprithms::memory::nest::DisjointRegions;

using Lower = std::vector<int64_t>;
using Upper = std::vector<int64_t>;

} // namespace alias
} // namespace memory
} // namespace poprithms

#endif
