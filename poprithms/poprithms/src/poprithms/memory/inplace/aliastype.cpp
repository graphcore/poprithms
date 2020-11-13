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
  case TypeEnum::Outplace: {
    ost << "Outplace";
    break;
  }
  case TypeEnum::AllInplace: {
    ost << "AllInplace";
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
  case TypeEnum::None: {
    ost << "None";
    break;
  }
  }
}

void AliasType::assertZeroOrOne(uint64_t index) {
  if (index != 0 && index != 1) {
    std::ostringstream oss;
    oss << "index=" << index << " is not valid, expected 0 or 1.";
    throw error(oss.str());
  }
}

} // namespace inplace
} // namespace memory
} // namespace poprithms
