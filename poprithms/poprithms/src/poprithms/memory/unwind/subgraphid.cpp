// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poprithms/memory/unwind/subgraphid.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

std::ostream &operator<<(std::ostream &ost, const SubGraphIds &ids) {
  append(ost, ids);
  return ost;
  // return vAppend(ost, ids);
}

} // namespace unwind
} // namespace memory
} // namespace poprithms
