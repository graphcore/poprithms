// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_USINGS_HPP
#define POPRITHMS_MEMORY_INPLACE_USINGS_HPP
#include "poprithms/memory/alias/graph.hpp"

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

} // namespace memory

namespace memory {
namespace inplace {

using Lower = alias::Lower;
using Upper = alias::Upper;

using DType = poprithms::ndarray::DType;

using InIndex  = poprithms::util::TypedInteger<'i', uint64_t>;
using OutIndex = poprithms::util::TypedInteger<'o', uint64_t>;
using OpId     = poprithms::util::TypedInteger<'O', int64_t>;

using Shape           = poprithms::ndarray::Shape;
using Permutation     = poprithms::util::Permutation;
using Region          = poprithms::memory::nest::Region;
using DisjointRegions = poprithms::memory::nest::DisjointRegions;

using Shapes = std::vector<Shape>;
using OpIds  = std::vector<OpId>;
std::ostream &operator<<(std::ostream &, const OpIds &);

using Constraint  = std::pair<OpId, OpId>;
using Constraints = std::vector<Constraint>;

class TensorId;
using TensorIds      = std::vector<TensorId>;
using ToAliasGraph   = std::vector<std::vector<alias::TensorId>>;
using FromAliasGraph = std::vector<TensorId>;

using Dimensions = poprithms::util::TypedVector<uint64_t, 'D', 'I', 'M', 'S'>;

using LowerPadding =
    util::TypedVector<uint64_t, 'L', 'o', 'w', 'P', 'a', 'd'>;

using UpperPadding =
    util::TypedVector<uint64_t, 'U', 'p', 'p', 'P', 'a', 'd'>;

using Strides =
    poprithms::util::TypedVector<int64_t, 'S', 't', 'r', 'i', 'd', 'e', 's'>;

static const alias::Color Constant = 0;
static const alias::Color Variable = 1;
using BroadcastPadding             = alias::BroadcastPadding;

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
