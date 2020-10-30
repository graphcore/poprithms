// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/memory/inplace/aliastype.hpp>
#include <poprithms/memory/inplace/error.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

std::ostream &operator<<(std::ostream &ost, AliasType t) {
  t.append(ost);
  return ost;
}

void AliasType::append(std::ostream &ost) const {
  switch (type_) {
  case TypeEnum::Out: {
    ost << "Outplace";
    break;
  }
  case TypeEnum::All: {
    ost << "All";
    break;
  }
  case TypeEnum::Binary0: {
    ost << "Binary0";
    break;
  }
  case TypeEnum::Binary1: {
    ost << "Binary1";
    break;
  }
  }
}

} // namespace inplace
} // namespace memory
} // namespace poprithms
