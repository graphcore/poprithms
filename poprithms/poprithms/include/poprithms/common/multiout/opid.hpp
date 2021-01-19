// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_MULTIOUT_OPID_HPP
#define POPRITHMS_COMMON_MULTIOUT_OPID_HPP
#include <ostream>
#include <vector>

#include <poprithms/util/typedinteger.hpp>

namespace poprithms {
namespace common {
namespace multiout {

using OpId  = poprithms::util::TypedInteger<'O', int64_t>;
using OpIds = std::vector<OpId>;
std::ostream &operator<<(std::ostream &, const OpIds &);

} // namespace multiout
} // namespace common
} // namespace poprithms

#endif
