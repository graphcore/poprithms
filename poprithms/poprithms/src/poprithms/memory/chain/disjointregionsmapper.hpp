// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_CHAIN_DISJOINTREGIONSMAPPER_HPP
#define POPRITHMS_MEMORY_CHAIN_DISJOINTREGIONSMAPPER_HPP
#include <numeric>
#include <sstream>
#include <variant>

#include <poprithms/memory/nest/region.hpp>

namespace poprithms {
namespace memory {
namespace chain {

using ndarray::Dimensions;
using ndarray::Shape;
using Lower     = ndarray::Shape::Lower;
using Upper     = ndarray::Shape::Upper;
using Stride    = ndarray::Stride;
using Dimension = ndarray::Dimension;
using nest::DisjointRegions;
using nest::Region;
using util::Permutation;

class DisjointRegionsMapper {
public:
  static DisjointRegions reshape(const DisjointRegions &x, const Shape &s) {
    return x.reshape(s);
  }

  static DisjointRegions expand(const DisjointRegions &x, const Shape &s) {
    return x.expand(s);
  }

  static DisjointRegions reduce(const DisjointRegions &x, const Shape &s) {
    return x.reduce(s);
  }

  static DisjointRegions settSample(const DisjointRegions &x,
                                    const Region &r) {
    return x.settSample(r);
  }

  static DisjointRegions settFillInto(const DisjointRegions &x,
                                      const Region &r) {
    return x.settFillInto(r);
  }

  static DisjointRegions reverse(const DisjointRegions &x,
                                 const Dimensions &d) {
    return x.reverse(d.get());
  }
  static DisjointRegions dimShuffle(const DisjointRegions &x,
                                    const Permutation &p) {
    return x.dimShuffle(p);
  }
};

} // namespace chain
} // namespace memory
} // namespace poprithms

#endif
