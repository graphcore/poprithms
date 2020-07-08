// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef PROTEA_TESTUTIL_MEMORY_NEST_RANDOMREGION_HPP
#define PROTEA_TESTUTIL_MEMORY_NEST_RANDOMREGION_HPP

#include <poprithms/memory/nest/region.hpp>

namespace poprithms {
namespace memory {
namespace nest {

/**
 * Generate a pair of Shapes, where the 2 Shapes have the same number of
 * elements. Examples might be {2,6,5} and {3,5,4}, two Shapes with 60
 * elements, or {1,1,2,5} and {10, 1}, two Shapes with 10 elements.
 *
 * \param seed The random seed.
 *
 * \param l0 The required length of the first Shape.
 *
 * \param l1 The required length of the second Shape.
 *
 * \param nDistinctFactors The number of prime factors (and 1) to choose from.
 *                         For example, if this is 4 then factors will be
 *                         drawn from {1,2,3,5}.
 *
 * \param nFactors The total number of factors drawn uniformally from the
 *                 pool, which compose the size of the Shape.
 *
 *
 * */
std::array<Shape, 2> getShapes(uint32_t seed,
                               uint64_t l0,
                               uint64_t l1,
                               uint64_t nDistinctFactors,
                               uint64_t nFactors);

Region getRandomRegion(const Shape &sh, uint32_t seed, uint64_t maxSettDepth);

} // namespace nest
} // namespace memory
} // namespace poprithms

#endif
