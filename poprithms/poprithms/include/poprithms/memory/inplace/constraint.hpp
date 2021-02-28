// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_CONSTRAINT_HPP
#define POPRITHMS_MEMORY_INPLACE_CONSTRAINT_HPP

#include <utility>
#include <vector>

#include <poprithms/common/multiout/opid.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

using common::multiout::OpId;
using common::multiout::OpIds;
using Constraint  = std::pair<OpId, OpId>;
using Constraints = std::vector<Constraint>;

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
