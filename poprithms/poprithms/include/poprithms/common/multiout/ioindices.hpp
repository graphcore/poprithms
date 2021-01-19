// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_MULTIOUT_USINGS_HPP
#define POPRITHMS_COMMON_MULTIOUT_USINGS_HPP
#include <vector>

#include <poprithms/util/typedinteger.hpp>

namespace poprithms {
namespace common {
namespace multiout {

using InIndex  = poprithms::util::TypedInteger<'i', uint64_t>;
using OutIndex = poprithms::util::TypedInteger<'o', uint64_t>;

using OutIndices = std::vector<OutIndex>;
using InIndices  = std::vector<InIndex>;

} // namespace multiout
} // namespace common
} // namespace poprithms

#endif
