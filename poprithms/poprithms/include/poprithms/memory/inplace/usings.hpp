// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_USINGS_HPP
#define POPRITHMS_MEMORY_INPLACE_USINGS_HPP
#include <ostream>
#include <utility>
#include <vector>

#include <poprithms/memory/alias/usings.hpp>
#include <poprithms/util/typedinteger.hpp>
#include <poprithms/util/typedvector.hpp>

namespace poprithms {

namespace util {
class Permutation;
}

namespace ndarray {
class Shape;
enum class DType;
} // namespace ndarray

namespace memory {
namespace nest {
class Region;
class DisjointRegions;
} // namespace nest

namespace alias {
enum class BroadcastPadding;
}

} // namespace memory

namespace memory {
namespace inplace {

using DType = poprithms::ndarray::DType;

using InIndex  = poprithms::util::TypedInteger<'i', uint64_t>;
using OutIndex = poprithms::util::TypedInteger<'o', uint64_t>;

using OpId  = poprithms::util::TypedInteger<'O', int64_t>;
using OpIds = std::vector<OpId>;
std::ostream &operator<<(std::ostream &, const OpIds &);

using GraphId = poprithms::util::TypedInteger<'g', uint32_t>;

using Shape  = poprithms::ndarray::Shape;
using Shapes = std::vector<Shape>;

using Permutation     = poprithms::util::Permutation;
using Region          = poprithms::memory::nest::Region;
using DisjointRegions = poprithms::memory::nest::DisjointRegions;

using Constraint  = std::pair<OpId, OpId>;
using Constraints = std::vector<Constraint>;

using Dimensions = poprithms::util::TypedVector<uint64_t, 'D', 'I', 'M', 'S'>;

using LowerPadding =
    util::TypedVector<uint64_t, 'L', 'o', 'w', 'P', 'a', 'd'>;

using UpperPadding =
    util::TypedVector<uint64_t, 'U', 'p', 'p', 'P', 'a', 'd'>;

using Strides =
    poprithms::util::TypedVector<int64_t, 'S', 't', 'r', 'i', 'd', 'e', 's'>;

static const alias::Color ConstantColor = 0;
static const alias::Color VariableColor = 1;

using Lower            = alias::Lower;
using Upper            = alias::Upper;
using BroadcastPadding = alias::BroadcastPadding;

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
