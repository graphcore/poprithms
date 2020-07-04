// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef PROTEA_TESTUTIL_ALIAS_RANDOMSETT_HPP
#define PROTEA_TESTUTIL_ALIAS_RANDOMSETT_HPP

#include <poprithms/memory/nest/sett.hpp>

namespace poprithms {
namespace memory {
namespace nest {
Sett getRandom(bool shorten,
               int64_t depth,
               bool canonicalize,
               int seed,
               int max0);
}
} // namespace memory
} // namespace poprithms

#endif
