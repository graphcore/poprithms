// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace common {
namespace schedulable {

std::ostream &operator<<(std::ostream &ost, const SubGraphIds &ids) {
  util::append(ost, ids);
  return ost;
}
std::ostream &operator<<(std::ostream &ost, const SubGraphId &id) {
  ost << id.get_u32();
  return ost;
}

} // namespace schedulable
} // namespace common
} // namespace poprithms
