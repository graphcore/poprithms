// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_ALIAS_ALIASUSINGS_HPP
#define POPRITHMS_MEMORY_ALIAS_ALIASUSINGS_HPP

#include <vector>

#include <poprithms/memory/gbase/gbaseusings.hpp>
#include <poprithms/util/typedinteger.hpp>

namespace poprithms {
namespace memory {

namespace alias {

using namespace gbaseusings;

using AllocId = poprithms::util::TypedInteger<'a', uint32_t>;
using Color   = poprithms::util::TypedInteger<'c', uint32_t>;

} // namespace alias
} // namespace memory
} // namespace poprithms

#endif
