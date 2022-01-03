// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <sstream>

#include <memory/unwind/error.hpp>

#include <poprithms/memory/unwind/sumlike.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

SumAttractions::SumAttractions(const SumAttractions::V &x, double defVal)
    : defaultValue_(defVal) {
  for (const auto &triplet : x) {
    const auto i0 = std::get<0>(triplet);
    const auto i1 = std::get<1>(triplet);
    if (i0 == i1) {
      throw error("InIndexes must be different for SumAttractions, repeated "
                  "InIndex " +
                  std::to_string(i0.get()));
    }
    const auto found = vs.find({i0, i1});
    if (found == vs.cend()) {
      vs.insert({{i0, i1}, std::get<2>(triplet)});
      vs.insert({{i1, i0}, std::get<2>(triplet)});
    } else {
      std::ostringstream oss;
      oss << "Duplicate attraction pairs in SumAttractions for InIndexes "
          << i0 << " and " << i1 << '.';
      throw error(oss.str());
    }
  }
}

double SumAttractions::get(InIndex i0, InIndex i1) const {
  auto found = vs.find({i0, i1});
  if (found != vs.cend()) {
    return found->second;
  }
  return defaultValue_;
}

} // namespace unwind
} // namespace memory
} // namespace poprithms
