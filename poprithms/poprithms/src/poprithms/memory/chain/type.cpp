// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <vector>

#include <memory/chain/error.hpp>

#include <poprithms/memory/chain/type.hpp>

namespace poprithms {
namespace memory {
namespace chain {

std::string getTypeString(Type t) {
  switch (t) {
  case Type::DimShuffle:
    return "DimShuffle";
  case Type::Expand:
    return "Expand";
  case Type::Reduce:
    return "Reduce";
  case Type::Reshape:
    return "Reshape";
  case Type::Reverse:
    return "Reverse";
  case Type::SettSample:
    return "SettSample";
  case Type::SettFillInto:
    return "SettFillInto";
  }
  throw error("unhandled case in getTypeString");
}

std::ostream &operator<<(std::ostream &ost, Type t) {
  ost << getTypeString(t);
  return ost;
}

Types TypeOrders::alphabetical() {
  return std::vector<Type>{Type::DimShuffle,
                           Type::Expand,
                           Type::Reduce,
                           Type::Reshape,
                           Type::Reverse,
                           Type::SettFillInto,
                           Type::SettSample};
}

Types TypeOrders::reverseAlphabetical() {
  auto x = alphabetical();
  std::reverse(x.begin(), x.end());
  return x;
}

} // namespace chain
} // namespace memory
} // namespace poprithms
